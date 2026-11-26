use std::{path::Path, time::{Duration, Instant}};
use burn::tensor::Device;
use crate::{audio, checkpoint::ModelPaths, AUDIO_SAMPLE_RATE, Note, Track, Transcription,
    drums::{DrumTranscriber, MIN_SOURCE_SAMPLE_RATE, DRUM_SAMPLE_RATE},
    evidence::{model::EvidenceNet, repair::{RepairCandidate, RepairInfo}},
    grouper::pipeline::Grouper,
    resample, retune::{self, Tuning},
    transcriber::{DecodedNote, Transcriber}};

#[derive(Clone, Debug)]
pub struct PipelineInfo {
    pub tuning: Tuning,
    pub notes: RepairInfo,
    pub grouped_notes: usize,
    pub drum_hits: usize,
    pub drums_skipped_low_sample_rate: bool,
}

#[derive(Clone, Debug, Default)]
pub struct PipelineTimings {
    pub preparation: Duration,
    pub transcriber: Option<Duration>,
    pub evidence: Duration,
    pub grouper: Duration,
    pub drums: Option<Duration>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DettatoModelPaths {
    pub transcriber: ModelPaths,
    pub evidence: ModelPaths,
    pub grouper: ModelPaths,
    pub drums: ModelPaths,
}

pub struct PipelineResult {
    pub transcription: Transcription,
    pub info: PipelineInfo,
    pub timings: PipelineTimings,
}

pub struct Dettato {
    pub transcriber: Transcriber,
    pub evidence: EvidenceNet,
    pub grouper: Grouper,
    pub drums: DrumTranscriber,
    resample_low_rate_drums: bool,
}

fn to_decoded(repaired: &[RepairCandidate]) -> Vec<DecodedNote> {
    repaired.iter().map(|note| DecodedNote {
        onset: note.onset,
        offset: note.offset.max(note.onset + 0.03),
        pitch: note.pitch,
        confidence: note.score,
        slot: 1,
    }).collect()
}

pub fn supports_drums(source_rate:u32)->bool { source_rate>=MIN_SOURCE_SAMPLE_RATE }

impl Dettato {
    pub fn from_directory(root: impl AsRef<Path>, device:&Device)->Result<Self,String>{
        let root = root.as_ref();
        Self::from_paths(&DettatoModelPaths {
            transcriber: ModelPaths::in_directory(root, "transcriber")?,
            evidence: ModelPaths::in_directory(root, "evidence")?,
            grouper: ModelPaths::in_directory(root, "grouper")?,
            drums: ModelPaths::in_directory(root, "drums")?,
        }, device)
    }

    pub fn from_paths(paths: &DettatoModelPaths, device: &Device) -> Result<Self, String> {
        Ok(Self {
            transcriber: Transcriber::from_paths(&paths.transcriber, device)?,
            evidence: EvidenceNet::from_paths(&paths.evidence, device)?,
            grouper: Grouper::from_paths(&paths.grouper, device)?,
            drums: DrumTranscriber::from_paths(&paths.drums, device)?,
            resample_low_rate_drums: false,
        })
    }

    pub fn transcribe_file(&self,path:impl AsRef<Path>)->Result<PipelineResult,String>{
        let path=path.as_ref();
        let decoded=audio::decode(path)?;
        self.transcribe_named(&path.to_string_lossy(),&decoded.samples,decoded.sample_rate)
    }

    pub fn set_merge_same_instrument(&mut self,merge:bool){
        self.grouper.decode_config.merge_same_instrument=merge;
    }

    /// Allow the drums model to use upsampled input below its native source-rate threshold.
    pub fn set_resample_low_rate_drums(&mut self, enabled: bool) {
        self.resample_low_rate_drums = enabled;
    }

    /// Transcribe mono floating-point PCM with the original source sample rate.
    /// Pitched stages use retuned 16 kHz data; Drums uses unretuned 44.1 kHz
    /// data. By default, recordings below 40 kHz skip the drums model.
    pub fn transcribe(&self,samples:&[f32],source_rate:u32)->Result<PipelineResult,String>{
        self.transcribe_named("<array>",samples,source_rate)
    }

    pub fn transcribe_named(&self,source:&str,samples:&[f32],source_rate:u32)->Result<PipelineResult,String>{
        if source_rate==0 || source_rate>384_000 || samples.is_empty() ||
            samples.iter().any(|v|!v.is_finite())
        {return Err("invalid source rate or PCM samples".into());}
        let seconds=samples.len() as f64/source_rate as f64;
        let preparation_started = Instant::now();
        let audio16=resample::resample(samples,source_rate,AUDIO_SAMPLE_RATE as f64)?;
        let (retuned,tuning)=retune::normalize(&audio16)?;
        let preparation = preparation_started.elapsed();
        let transcriber_started = Instant::now();
        let shared = self.transcriber.shared_run(&retuned)?;
        let transcriber = shared.as_ref().map(|_| transcriber_started.elapsed());
        let evidence_started = Instant::now();
        let (repaired, notes) = if let Some(ref run) = shared {
            let (decoded, evidence) = run.notes_with_evidence(self.transcriber.decoder_config())?;
            self.evidence.repair_notes(&decoded, &evidence)?
        } else {
            self.evidence.transcribe_repaired(&self.transcriber, &retuned)?
        };
        let evidence = evidence_started.elapsed();
        let grouper_started = Instant::now();
        let mut tracks = if repaired.is_empty() {
            Vec::new()
        } else if let Some(ref run) = shared {
            let (meta, taps) = run.grouper_features(&to_decoded(&repaired), self.transcriber.decoder_config())?;
            self.grouper.group(&retuned, &meta, &taps, tuning.time_scale)?
        } else {
            self.grouper.group_with_transcriber(&retuned, &to_decoded(&repaired), &self.transcriber, tuning.time_scale)?
        };
        let grouper = grouper_started.elapsed();
        drop(shared);
        let grouped_notes=tracks.iter().map(|t|t.notes.len()).sum();
        let mut drum_hits=0;
        let run_drums = supports_drums(source_rate) || self.resample_low_rate_drums;
        let drums = if run_drums {
            let started = Instant::now();
            let audio44=resample::resample(samples,source_rate,DRUM_SAMPLE_RATE as f64)?;
            if let Some(out)=self.drums.transcribe(&audio44)? {
                drum_hits=out.hits.len();
                if !out.hits.is_empty() {
                    let notes=out.hits.into_iter().map(|n|
                        Note::new(n.onset,n.offset,n.pitch,n.velocity)).collect::<Result<Vec<_>,_>>()?;
                    tracks.push(Track{name:"Drums".into(),program:0,is_drum:true,notes});
                }
            }
            Some(started.elapsed())
        } else {
            None
        };
        Ok(PipelineResult{
            transcription:Transcription{source:source.to_string(),seconds,tracks},
            timings: PipelineTimings { preparation, transcriber, evidence, grouper, drums },
            info:PipelineInfo{
                tuning,notes,grouped_notes,drum_hits,
                drums_skipped_low_sample_rate:!run_drums,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::repair::RepairKind;
    #[test]
    fn drum_gate_uses_original_rate_not_resampled_rate(){
        assert!(!supports_drums(16000));
        assert!(!supports_drums(39999));
        assert!(supports_drums(40000));
        assert!(supports_drums(44100));
    }
    #[test]
    fn repaired_notes_have_minimum_duration(){
        let notes=to_decoded(&[RepairCandidate{
            onset:0.5,offset:0.51,pitch:60,score:0.85,kind:RepairKind::Inserted,
        }]);
        assert_eq!(notes.len(),1);
        assert!((notes[0].offset-0.53).abs()<1e-10);
    }
}
