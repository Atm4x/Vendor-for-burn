pub mod checkpoint;
pub mod audio;
pub mod evidence;
pub mod grouper;
pub mod drums;
pub mod hcqt;
pub mod midi;
pub mod resample;
pub mod retune;
pub mod pipeline;
pub mod transcriber;

pub const AUDIO_SAMPLE_RATE: usize = 16_000;
pub const HOP_SAMPLES: usize = 160;
pub const PITCH_LO: u8 = 21;
pub const PITCH_COUNT: usize = 88;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Note {
    pub onset: f64,
    pub offset: f64,
    pub pitch: u8,
    pub velocity: u8,
}

impl Note {
    pub fn new(onset: f64, offset: f64, pitch: u8, velocity: u8) -> Result<Self, String> {
        if !onset.is_finite() || !offset.is_finite() || onset < 0.0 || offset <= onset {
            return Err("note onset/offset must be finite, nonnegative and ordered".into());
        }
        if pitch > 127 || !(1..=127).contains(&velocity) {
            return Err("invalid MIDI pitch or velocity".into());
        }
        Ok(Self { onset, offset, pitch, velocity })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub name: String,
    pub program: u8,
    pub is_drum: bool,
    pub notes: Vec<Note>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Transcription {
    pub source: String,
    pub seconds: f64,
    pub tracks: Vec<Track>,
}
