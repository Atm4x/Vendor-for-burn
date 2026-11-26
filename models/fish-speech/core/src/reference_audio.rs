use crate::audio::{firefly_reference_lengths, FireflyFrameLengths, N_MELS, SAMPLE_RATE};
use crate::error::{FishError, Result};
use crate::firefly_cpu::AudioTensor;
use crate::firefly_ir::FireflyGraph;
use crate::spectrogram_cpu::FishLogMel;

#[derive(Debug, Clone)]
pub struct ReferenceMelInput {
    pub mels: AudioTensor,
    pub lengths: FireflyFrameLengths,
}

impl ReferenceMelInput {
    pub fn network_shape(&self) -> [usize; 4] {
        [self.mels.batch, self.mels.channels, 1, self.mels.frames]
    }

    pub fn reference_graph(&self) -> Result<FireflyGraph> {
        FireflyGraph::reference_encode(self.mels.batch, self.lengths.input_audio_samples)
    }

    pub fn validate(&self) -> Result<()> {
        let graph = self.reference_graph()?;
        let mel = graph
            .nodes
            .iter()
            .find(|n| n.name == "spec_transform.logmel")
            .ok_or(FishError::InvalidShape("missing reference LogMel boundary"))?;
        if self.mels.channels != N_MELS
            || self.mels.frames != self.lengths.mel_frames
            || self.network_shape() != [mel.output.batch, mel.output.channels, 1, mel.output.frames]
            || self.lengths != firefly_reference_lengths(self.lengths.input_audio_samples)?
        {
            return Err(FishError::InvalidShape("reference mels and graph disagree"));
        }
        Ok(())
    }
}

pub struct ReferenceMelPreprocessor {
    transform: FishLogMel,
}

impl ReferenceMelPreprocessor {
    pub fn new() -> Self {
        Self {
            transform: FishLogMel::new(),
        }
    }

    pub fn prepare_mono(&self, pcm: &[&[f32]], sample_rate: usize) -> Result<ReferenceMelInput> {
        if sample_rate != SAMPLE_RATE {
            return Err(FishError::InvalidConfig(
                "reference PCM must be mono 44100 Hz; resample before preprocessing",
            ));
        }
        let first = pcm
            .first()
            .ok_or(FishError::InvalidShape("empty reference audio batch"))?;
        if pcm.iter().any(|samples| samples.len() != first.len()) {
            return Err(FishError::InvalidShape(
                "reference batch requires uniform audio lengths",
            ));
        }
        let lengths = firefly_reference_lengths(first.len())?;
        let mut data = Vec::with_capacity(pcm.len() * N_MELS * lengths.mel_frames);
        for samples in pcm {
            let mel = self.transform.transform_mono(samples)?;
            if mel.frames != lengths.mel_frames || mel.channels != N_MELS {
                return Err(FishError::InvalidShape(
                    "LogMel output differs from graph profile",
                ));
            }
            data.extend_from_slice(&mel.data);
        }
        let input = ReferenceMelInput {
            mels: AudioTensor::new(pcm.len(), N_MELS, lengths.mel_frames, data)?,
            lengths,
        };
        input.validate()?;
        Ok(input)
    }
}

impl Default for ReferenceMelPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_matches_independent_reference_mel_and_graph() {
        let first = (0..3073)
            .map(|i| (i as f32 / 47.0).sin())
            .collect::<Vec<_>>();
        let second = (0..3073)
            .map(|i| (i as f32 / 17.0).cos())
            .collect::<Vec<_>>();
        let preprocessor = ReferenceMelPreprocessor::new();
        let batch = preprocessor
            .prepare_mono(&[&first, &second], 44100)
            .unwrap();
        assert_eq!(batch.network_shape(), [2, 160, 1, 6]);
        assert_eq!(batch.lengths.valid_code_frames, 1);
        let first_mel = FishLogMel::new().transform_mono(&first).unwrap();
        assert_eq!(&batch.mels.data[..first_mel.data.len()], first_mel.data);
        assert_eq!(
            batch
                .reference_graph()
                .unwrap()
                .nodes
                .last()
                .unwrap()
                .output
                .channels,
            8
        );
    }

    #[test]
    fn unequal_audio_lengths_and_bad_rate_fail_explicitly() {
        let preprocessor = ReferenceMelPreprocessor::new();
        let first = vec![0.0; 4096];
        let second = vec![0.0; 4095];
        assert!(preprocessor
            .prepare_mono(&[&first, &second], 44100)
            .is_err());
        assert!(preprocessor.prepare_mono(&[&first], 16000).is_err());
        assert!(preprocessor.prepare_mono(&[], 44100).is_err());
        assert!(preprocessor.prepare_mono(&[&first[..768]], 44100).is_err());
    }
}
