//! Frame-domain speech editing from the original F5-TTS speech_edit.py.

use crate::{
    F5Error, Result,
    mel::{HOP_SIZE, MEL_CHANNELS, SAMPLE_RATE},
};

const MAX_FRAMES: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EditSpan {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub replacement_seconds: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechEditPlan {
    pub conditioning_mel: Vec<f32>,
    pub keep_frames: Vec<bool>,
    pub original_frames: usize,
    pub edited_frames: usize,
    pub total_frames: usize,
}

fn seconds_to_frames(seconds: f64) -> Result<usize> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(F5Error::InvalidConfiguration(
            "Edit times must be finite and nonnegative",
        ));
    }
    let frames = (seconds * SAMPLE_RATE as f64 / HOP_SIZE as f64).round_ties_even();
    if frames > MAX_FRAMES as f64 {
        return Err(F5Error::InvalidConfiguration(
            "Edit timeline exceeds CFM frame limit",
        ));
    }
    Ok(frames as usize)
}

impl SpeechEditPlan {
    /// Builds the original Python mel-domain edit condition and frame mask.
    /// Original mel is extracted before any frames are replaced, so STFT windows
    /// at the edit boundaries never straddle injected zeros.
    pub fn new(
        original_mel: &[f32],
        original_frames: usize,
        spans: &[EditSpan],
        text_token_count: usize,
    ) -> Result<Self> {
        if original_frames == 0
            || original_frames >= MAX_FRAMES
            || text_token_count == 0
            || text_token_count > 8192
            || spans.is_empty()
        {
            return Err(F5Error::InvalidConfiguration(
                "Invalid speech edit input lengths",
            ));
        }
        if original_frames.checked_mul(MEL_CHANNELS) != Some(original_mel.len())
            || original_mel.iter().any(|v| !v.is_finite())
        {
            return Err(F5Error::InvalidShape(
                "Original mel must be finite [frames, 100]",
            ));
        }

        let mut kept_or_inserted = Vec::with_capacity(original_mel.len());
        let mut keep = Vec::with_capacity(original_frames);
        let mut offset_frame = 0;
        for span in spans {
            if span.end_seconds <= span.start_seconds {
                return Err(F5Error::InvalidConfiguration(
                    "Speech edit spans must have positive duration",
                ));
            }
            let start = seconds_to_frames(span.start_seconds)?;
            let end = seconds_to_frames(span.end_seconds)?;
            let seconds = span
                .replacement_seconds
                .unwrap_or(span.end_seconds - span.start_seconds);
            let replacement = seconds_to_frames(seconds)?;
            if start < offset_frame || end > original_frames || start >= end || replacement == 0 {
                return Err(F5Error::InvalidConfiguration(
                    "Speech edit spans must be ordered, disjoint, inside audio and nonempty",
                ));
            }
            let prefix = start - offset_frame;
            kept_or_inserted.extend_from_slice(
                &original_mel[offset_frame * MEL_CHANNELS..start * MEL_CHANNELS],
            );
            kept_or_inserted.resize(
                kept_or_inserted
                    .len()
                    .checked_add(replacement * MEL_CHANNELS)
                    .ok_or(F5Error::InvalidShape("Speech edit mel length overflow"))?,
                0.0,
            );
            keep.extend(std::iter::repeat_n(true, prefix));
            keep.extend(std::iter::repeat_n(false, replacement));
            offset_frame = end;
        }
        let suffix = original_frames - offset_frame;
        kept_or_inserted.extend_from_slice(&original_mel[offset_frame * MEL_CHANNELS..]);
        keep.extend(std::iter::repeat_n(true, suffix));
        let edited_frames = keep.len();
        if edited_frames >= MAX_FRAMES {
            return Err(F5Error::InvalidConfiguration(
                "Speech edit output exceeds frame limit",
            ));
        }
        // CFM.sample enforces max(text_tokens, cond_frames) + 1, even for speech editing.
        let total_frames = edited_frames
            .max(text_token_count)
            .checked_add(1)
            .ok_or(F5Error::InvalidShape("Speech edit duration overflow"))?;
        if total_frames > MAX_FRAMES {
            return Err(F5Error::InvalidConfiguration(
                "Speech edit output exceeds frame limit",
            ));
        }
        kept_or_inserted.resize(total_frames * MEL_CHANNELS, 0.0);
        keep.resize(total_frames, false);
        Ok(Self {
            conditioning_mel: kept_or_inserted,
            keep_frames: keep,
            original_frames,
            edited_frames,
            total_frames,
        })
    }

    pub fn restore(&self, generated_mel: &[f32]) -> Result<Vec<f32>> {
        if generated_mel.len() != self.conditioning_mel.len() {
            return Err(F5Error::InvalidShape(
                "Generated speech edit mel has incorrect dimensions",
            ));
        }
        let mut out = generated_mel.to_vec();
        for (frame, &keep) in self.keep_frames.iter().enumerate() {
            if keep {
                let start = frame * MEL_CHANNELS;
                out[start..start + MEL_CHANNELS]
                    .copy_from_slice(&self.conditioning_mel[start..start + MEL_CHANNELS]);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_mask_and_mel_match_upstream_speech_edit_python_ast() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/speech_edit_upstream_expected.json"
        ))
        .unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 320);
        for (index, case) in cases.iter().enumerate() {
            let frames = case["original_frames"].as_u64().unwrap() as usize;
            let original: Vec<f32> = (0..frames * MEL_CHANNELS).map(|v| v as f32).collect();
            let spans: Vec<EditSpan> = case["spans"]
                .as_array()
                .unwrap()
                .iter()
                .map(|span| EditSpan {
                    start_seconds: f64::from_bits(span["start_bits"].as_u64().unwrap()),
                    end_seconds: f64::from_bits(span["end_bits"].as_u64().unwrap()),
                    replacement_seconds: span["replacement_bits"].as_u64().map(f64::from_bits),
                })
                .collect();
            let text_tokens = case["text_tokens"].as_u64().unwrap() as usize;
            let plan = SpeechEditPlan::new(&original, frames, &spans, text_tokens)
                .unwrap_or_else(|error| panic!("case {index}: {error}"));
            assert_eq!(
                plan.edited_frames,
                case["edited_frames"].as_u64().unwrap() as usize,
                "case {index} edited frames"
            );
            assert_eq!(
                plan.total_frames,
                case["total_frames"].as_u64().unwrap() as usize,
                "case {index} total frames"
            );
            let mask: String = plan
                .keep_frames
                .iter()
                .map(|&b| if b { '1' } else { '0' })
                .collect();
            assert_eq!(mask, case["mask"].as_str().unwrap(), "case {index} mask");
            let mut hash = 0xcbf29ce484222325_u64;
            for value in &plan.conditioning_mel {
                for byte in value.to_le_bytes() {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
                }
            }
            assert_eq!(
                format!("{hash:016x}"),
                case["mel_fnv64"].as_str().unwrap(),
                "case {index} mel content"
            );
        }
    }

    #[test]
    fn editing_replaces_multiple_disjoint_regions_and_keeps_original_mel() {
        let original: Vec<f32> = (0..32 * MEL_CHANNELS).map(|v| v as f32).collect();
        let to_sec = |frame: usize| frame as f64 * HOP_SIZE as f64 / SAMPLE_RATE as f64;
        let plan = SpeechEditPlan::new(
            &original,
            32,
            &[
                EditSpan {
                    start_seconds: to_sec(3),
                    end_seconds: to_sec(8),
                    replacement_seconds: Some(to_sec(7)),
                },
                EditSpan {
                    start_seconds: to_sec(15),
                    end_seconds: to_sec(22),
                    replacement_seconds: Some(to_sec(4)),
                },
            ],
            10,
        )
        .unwrap();
        assert_eq!(plan.edited_frames, 31);
        assert_eq!(plan.total_frames, 32);
        assert_eq!(&plan.keep_frames[..3], &[true; 3]);
        assert_eq!(&plan.keep_frames[3..10], &[false; 7]);
        assert_eq!(&plan.keep_frames[10..17], &[true; 7]);
        assert_eq!(&plan.keep_frames[17..21], &[false; 4]);
        assert_eq!(&plan.keep_frames[21..31], &[true; 10]);
        assert!(!plan.keep_frames[31]);
        let generated = vec![42.0; plan.total_frames * MEL_CHANNELS];
        let restored = plan.restore(&generated).unwrap();
        assert_eq!(&restored[0..300], &original[0..300]);
        assert_eq!(&restored[300..1000], &vec![42.0; 700]);
        assert_eq!(&restored[1000..1700], &original[800..1500]);
    }

    #[test]
    fn speech_edit_rejects_overlaps_nonfinite_and_out_of_range() {
        let original = vec![1.0; 2000];
        let mut span = EditSpan {
            start_seconds: 0.03,
            end_seconds: 0.09,
            replacement_seconds: None,
        };
        assert!(SpeechEditPlan::new(&original, 20, &[span], 8).is_ok());
        span.replacement_seconds = Some(f64::NAN);
        assert!(SpeechEditPlan::new(&original, 20, &[span], 8).is_err());
        span.replacement_seconds = None;
        assert!(SpeechEditPlan::new(&original, 20, &[span, span], 8).is_err());
        span.end_seconds = 1.0;
        assert!(SpeechEditPlan::new(&original, 20, &[span], 8).is_err());
    }
}
