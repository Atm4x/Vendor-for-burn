use super::{
    decode::GroupConfig,
    naming::TrackHead,
    net::{NoteNet, NoteNetConfig, TRANSCRIBER_TAP_DIM},
    render::render_tracks,
};
use crate::{Track, checkpoint::{Checkpoint, ModelPaths}, transcriber::{DecodedNote, Transcriber}};
use burn::tensor::Device;
use serde_json::Value;
use std::path::Path;

pub struct Grouper {
    pub network: NoteNet,
    pub decode_config: GroupConfig,
    track_head: TrackHead,
}

impl Grouper {
    pub fn from_directory(root: impl AsRef<Path>, device: &Device) -> Result<Self, String> {
        let paths = ModelPaths::in_directory(root, "grouper")?;
        let json = paths.load_config()?;
        let checkpoint = Checkpoint::open(&paths.weights)?;
        let config = NoteNetConfig::from_json(&json)?;
        let network = NoteNet::load(&checkpoint, device, config)?;
        let track_head = TrackHead::load(&checkpoint)?;
        let decode_config = Self::parse_decode_config(&json)?;
        Ok(Self { network, track_head, decode_config })
    }

    pub fn parse_decode_config(json: &Value) -> Result<GroupConfig, String> {
        let mut config = GroupConfig::default();
        let Some(decode) = json.get("decode") else { return Ok(config); };
        let f = |key: &str, default: f64| -> Result<f64, String> {
            decode.get(key).map(|v| v.as_f64().ok_or_else(|| format!("invalid Grouper decode.{key}")))
                .unwrap_or(Ok(default))
        };
        let u = |key: &str, default: usize| -> Result<usize, String> {
            decode.get(key).map(|v| v.as_u64().map(|x| x as usize)
                .ok_or_else(|| format!("invalid Grouper decode.{key}"))).unwrap_or(Ok(default))
        };
        let b = |key: &str, default: bool| -> Result<bool, String> {
            decode.get(key).map(|v| v.as_bool().ok_or_else(|| format!("invalid Grouper decode.{key}")))
                .unwrap_or(Ok(default))
        };
        config.tau = f("tau", config.tau)?;
        config.tau2 = f("tau2", config.tau2)?;
        config.lam = f("lam", config.lam)?;
        config.min_notes = u("min_notes", config.min_notes)?;
        config.refine = u("refine", config.refine)?;
        config.alpha = f("alpha", config.alpha)?;
        config.expected_credit = b("ec", config.expected_credit)?;
        config.merge_same_instrument = b("merge", config.merge_same_instrument)?;
        config.absorb = f("absorb", config.absorb)?;
        config.absorb_fraction = f("absorb_frac", config.absorb_fraction)?;
        Ok(config)
    }

    /// Run the complete Grouper stage from 12 note scalars and three Transcriber taps.
    /// `time_scale` restores original timestamps after optional retuning.
    pub fn group(
        &self, audio16k: &[f32],
        meta: &[[f32; 12]], taps: &[[[f32; TRANSCRIBER_TAP_DIM]; 3]],
        time_scale: f64,
    ) -> Result<Vec<Track>, String> {
        if meta.len() != taps.len() { return Err("Grouper metadata/taps are misaligned".into()); }
        if meta.is_empty() { return Ok(vec![]); }
        let predictions = self.network.predict_song(audio16k, meta, taps)?;
        let notes = meta.iter().map(|m| (f64::from(m[0]), f64::from(m[1]), m[2].round().clamp(0.0, 127.0) as u8))
            .collect::<Vec<_>>();
        let confidence = meta.iter().map(|m| if m[3] > 0.5 { m[5] } else { m[4] })
            .collect::<Vec<_>>();
        let groups = self.track_head.decode(&predictions, &notes, self.decode_config)?;
        render_tracks(&groups, &notes, &confidence, time_scale, !self.decode_config.merge_same_instrument)
    }

    /// Assemble the full input from Transcriber's output maps and internal taps.
    /// Does not rerun note detection: callers can pass Evidence-repaired notes.
    pub fn group_with_transcriber(
        &self, audio16k: &[f32], notes: &[DecodedNote], transcriber: &Transcriber,
        time_scale: f64,
    ) -> Result<Vec<Track>, String> {
        let (meta, taps) = transcriber.grouper_features(audio16k, notes)?;
        self.group(audio16k, &meta, &taps, time_scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_configuration_reproduces_python_keyword_names() {
        let json = serde_json::json!({"decode": {
            "tau":0.17,"tau2":0.3,"lam":0.8,"min_notes":5,
            "refine":4,"alpha":0.7,"ec":false,"merge":false,
            "absorb":0.2,"absorb_frac":0.15
        }});
        let cfg = Grouper::parse_decode_config(&json).unwrap();
        assert!((cfg.tau-0.17).abs()<1e-9);
        assert!(!cfg.expected_credit && !cfg.merge_same_instrument);
        assert_eq!(cfg.min_notes, 5);
        assert!((cfg.absorb_fraction-0.15).abs()<1e-9);
    }
}
