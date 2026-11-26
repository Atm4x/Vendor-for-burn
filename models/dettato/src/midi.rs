use crate::{Track, Transcription};
use std::{fs, io::Write, path::{Path, PathBuf}};

const TICKS_PER_QUARTER: u16 = 480;
const TICKS_PER_SECOND: f64 = 960.0;

#[derive(Clone)]
struct Event {
    tick: u32,
    priority: u8,
    bytes: Vec<u8>,
}

fn variable_length(mut value: u32, output: &mut Vec<u8>) {
    let mut pending = [0u8; 5];
    let mut length = 1;
    pending[0] = (value & 0x7f) as u8;
    while value > 0x7f {
        value >>= 7;
        pending[length] = ((value & 0x7f) as u8) | 0x80;
        length += 1;
    }
    for byte in pending[..length].iter().rev() { output.push(*byte); }
}

fn metadata(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0xff, kind];
    variable_length(payload.len() as u32, &mut bytes);
    bytes.extend_from_slice(payload);
    bytes
}

fn ticks(seconds: f64) -> Result<u32, String> {
    if !seconds.is_finite() || seconds < 0.0 || seconds * TICKS_PER_SECOND >= u32::MAX as f64 {
        return Err("invalid or excessive MIDI event time".into());
    }
    Ok((seconds * TICKS_PER_SECOND).round() as u32)
}

fn track_chunk(events: &mut Vec<Event>) -> Result<Vec<u8>, String> {
    events.sort_by(|a, b| a.tick.cmp(&b.tick).then_with(|| a.priority.cmp(&b.priority)));
    let mut body = Vec::new();
    let mut previous = 0;
    for event in events {
        variable_length(event.tick - previous, &mut body);
        body.extend_from_slice(&event.bytes);
        previous = event.tick;
    }
    body.extend_from_slice(&[0x00, 0xff, 0x2f, 0x00]);
    let mut result = b"MTrk".to_vec();
    result.extend_from_slice(&(u32::try_from(body.len()).map_err(|e| e.to_string())?).to_be_bytes());
    result.extend_from_slice(&body);
    Ok(result)
}

fn instrument_track(track: &Track, channel: u8, port: u8) -> Result<Vec<u8>, String> {
    if track.program > 127 { return Err(format!("{}: invalid program", track.name)); }
    let mut events = vec![
        Event { tick: 0, priority: 0, bytes: metadata(0x03, track.name.as_bytes()) },
        Event { tick: 0, priority: 1, bytes: metadata(0x21, &[port]) },
    ];
    if !track.is_drum {
        events.push(Event { tick: 0, priority: 2, bytes: vec![0xc0 | channel, track.program] });
    }
    for note in &track.notes {
        if note.pitch > 127 || !(1..=127).contains(&note.velocity)
            || !note.onset.is_finite() || !note.offset.is_finite() || note.offset <= note.onset
        {
            return Err(format!("{}: invalid note", track.name));
        }
        let on = ticks(note.onset)?;
        let off = ticks(note.offset)?.max(on.saturating_add(1));
        events.push(Event { tick: on, priority: 4, bytes: vec![0x90 | channel, note.pitch, note.velocity] });
        events.push(Event { tick: off, priority: 3, bytes: vec![0x80 | channel, note.pitch, 0] });
    }
    track_chunk(&mut events)
}

impl Transcription {
    pub fn to_midi_bytes(&self) -> Result<Vec<u8>, String> {
        self.encode_midi(&self.tracks)
    }

    pub fn track_midi_bytes(&self, index: usize) -> Result<Vec<u8>, String> {
        let track = self.tracks.get(index).ok_or("MIDI track index is out of range")?;
        self.encode_midi(std::slice::from_ref(track))
    }

    fn encode_midi(&self, tracks: &[Track]) -> Result<Vec<u8>, String> {
        if !self.seconds.is_finite() || self.seconds < 0.0 || tracks.len() > u16::MAX as usize - 1 {
            return Err("invalid MIDI transcription size".into());
        }
        let mut bytes = b"MThd".to_vec();
        bytes.extend_from_slice(&6u32.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&((tracks.len() + 1) as u16).to_be_bytes());
        bytes.extend_from_slice(&TICKS_PER_QUARTER.to_be_bytes());
        let mut tempo = vec![
            Event { tick: 0, priority: 0, bytes: metadata(0x03, b"Dettato") },
            Event { tick: 0, priority: 1, bytes: metadata(0x51, &[0x07, 0xa1, 0x20]) },
        ];
        bytes.extend_from_slice(&track_chunk(&mut tempo)?);
        let mut pitched = 0usize;
        for track in tracks {
            let (channel, port) = if track.is_drum {
                (9u8, 0u8)
            } else {
                let port = pitched / 15;
                if port > 127 { return Err("too many MIDI instrument ports".into()); }
                let index = (pitched % 15) as u8;
                pitched += 1;
                (if index >= 9 { index + 1 } else { index }, port as u8)
            };
            bytes.extend_from_slice(&instrument_track(track, channel, port)?);
        }
        Ok(bytes)
    }

    pub fn save_midi(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let midi = self.to_midi_bytes()?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        fs::write(path, midi).map_err(|e| e.to_string())
    }

    /// Export one MIDI file per track, with the original note timing and instrument program.
    pub fn save_midi_tracks(&self, directory: impl AsRef<Path>) -> Result<Vec<PathBuf>, String> {
        let directory = directory.as_ref();
        let exports = self.tracks.iter().enumerate().map(|(index, track)| {
            let name = track.name.split(|ch: char| !ch.is_ascii_alphanumeric())
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>().join("-").to_ascii_lowercase();
            let name = if name.is_empty() { "track" } else { &name };
            let path = directory.join(format!("{:02}-{name}.mid", index + 1));
            Ok((path, self.encode_midi(std::slice::from_ref(track))?))
        }).collect::<Result<Vec<_>, String>>()?;
        fs::create_dir_all(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
        let mut written = Vec::with_capacity(exports.len());
        for (path, midi) in exports {
            let result = fs::File::create_new(&path).and_then(|mut file| {
                written.push(path.clone());
                file.write_all(&midi)
            });
            if let Err(error) = result {
                for created in written {
                    let _ = fs::remove_file(created);
                }
                return Err(format!("{}: {error}", path.display()));
            }
        }
        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Note;
    #[test]
    fn standard_midi_file_preserves_multiple_program_tracks() {
        let one = Track {
            name: "Piano".into(), program: 0, is_drum: false,
            notes: vec![Note::new(0.1, 0.4, 60, 100).unwrap()],
        };
        let two = Track {
            name: "Bass".into(), program: 33, is_drum: false,
            notes: vec![Note::new(0.0, 0.7, 40, 97).unwrap()],
        };
        let song = Transcription { source: "x".into(), seconds: 1.0, tracks: vec![one, two] };
        let midi = song.to_midi_bytes().unwrap();
        assert_eq!(&midi[..4], b"MThd");
        assert_eq!(u16::from_be_bytes([midi[10], midi[11]]), 3);
        assert!(midi.windows(5).any(|w| w == b"Piano"));
        assert!(midi.windows(4).any(|w| w == b"Bass"));
        assert!(midi.windows(2).any(|w| w == [0xc1, 33]));
        assert!(midi.windows(3).any(|w| w == [0x99, 60, 100]) == false);
    }
    #[test]
    fn save_midi_creates_missing_parent_directories() {
        let root = std::env::temp_dir().join(format!("dettato-midi-test-{}", std::process::id()));
        let path = root.join("nested").join("result.mid");
        let song = Transcription {source: "synthetic".into(), seconds: 0.2, tracks: vec![]};
        song.save_midi(&path).unwrap();
        assert_eq!(&fs::read(&path).unwrap()[..4], b"MThd");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn note_off_precedes_retrigger_at_same_tick() {
        let track = Track {
            name: "Test".into(), program: 0, is_drum: false,
            notes: vec![Note::new(0.0, 0.2, 60, 100).unwrap(), Note::new(0.2, 0.4, 60, 99).unwrap()],
        };
        let song = Transcription { source: "test".into(), seconds: 1.0, tracks: vec![track] };
        let midi = song.to_midi_bytes().unwrap();
        let off = midi.windows(3).position(|w| w == [0x80, 60, 0]).unwrap();
        let second = midi.windows(3).position(|w| w == [0x90, 60, 99]).unwrap();
        assert!(off < second);
    }
}
