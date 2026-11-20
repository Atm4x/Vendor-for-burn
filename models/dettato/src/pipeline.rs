use std::path::Path;
use burn::tensor::Device;
use crate::{audio, AUDIO_SAMPLE_RATE, Note, Track, Transcription,
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

pub struct PipelineResult {
    pub transcription: Transcription,
    pub info: PipelineInfo,
}

pub struct Dettato {
    pub transcriber: Transcriber,
    pub evidence: EvidenceNet,
    pub grouper: Grouper,
    pub drums: DrumTranscriber,
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
        let root=root.as_ref();
        Ok(Self{
            transcriber:Transcriber::from_directory(root,device)?,
            evidence:EvidenceNet::from_directory(root,device)?,
            grouper:Grouper::from_directory(root,device)?,
            drums:DrumTranscriber::from_directory(root,device)?,
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

    /// Transcribe mono floating-point PCM with the original source sample rate.
    /// Pitched stages use retuned 16 kHz data; Drums uses unretuned 44.1 kHz
    /// data and is disabled for original recordings below 40 kHz.
    pub fn transcribe(&self,samples:&[f32],source_rate:u32)->Result<PipelineResult,String>{
        self.transcribe_named("<array>",samples,source_rate)
    }

    pub fn transcribe_named(&self,source:&str,samples:&[f32],source_rate:u32)->Result<PipelineResult,String>{
        if source_rate==0 || source_rate>384_000 || samples.is_empty() ||
            samples.iter().any(|v|!v.is_finite())
        {return Err("invalid source rate or PCM samples".into());}
        let seconds=samples.len() as f64/source_rate as f64;
        let audio16=resample::resample(samples,source_rate,AUDIO_SAMPLE_RATE as f64)?;
        let (retuned,tuning)=retune::normalize(&audio16)?;
        let (repaired,notes)=self.evidence.transcribe_repaired(&self.transcriber,&retuned)?;
        let mut tracks=if repaired.is_empty(){Vec::new()}else{
            self.grouper.group_with_transcriber(&retuned,&to_decoded(&repaired),&self.transcriber,tuning.time_scale)?
        };
        let grouped_notes=tracks.iter().map(|t|t.notes.len()).sum();
        let mut drum_hits=0;
        if supports_drums(source_rate) {
            let audio44=resample::resample(samples,source_rate,DRUM_SAMPLE_RATE as f64)?;
            if let Some(out)=self.drums.transcribe(&audio44)? {
                drum_hits=out.hits.len();
                if !out.hits.is_empty() {
                    let notes=out.hits.into_iter().map(|n|
                        Note::new(n.onset,n.offset,n.pitch,n.velocity)).collect::<Result<Vec<_>,_>>()?;
                    tracks.push(Track{name:"Drums".into(),program:0,is_drum:true,notes});
                }
            }
        }
        Ok(PipelineResult{
            transcription:Transcription{source:source.to_string(),seconds,tracks},
            info:PipelineInfo{
                tuning,notes,grouped_notes,drum_hits,
                drums_skipped_low_sample_rate:!supports_drums(source_rate),
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
