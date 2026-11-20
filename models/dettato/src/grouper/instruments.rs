#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instrument {
    pub name: &'static str,
    pub program: u8,
    pub family_mask: u32,
}

pub const INSTRUMENTS: [Instrument; 73] = [
    Instrument { name: "piano", program: 0, family_mask: 8 },
    Instrument { name: "electric_piano", program: 4, family_mask: 8 },
    Instrument { name: "harpsichord", program: 6, family_mask: 8 },
    Instrument { name: "clavinet", program: 7, family_mask: 8 },
    Instrument { name: "harp", program: 46, family_mask: 136 },
    Instrument { name: "celesta", program: 8, family_mask: 24 },
    Instrument { name: "glockenspiel", program: 9, family_mask: 16 },
    Instrument { name: "music_box", program: 10, family_mask: 16 },
    Instrument { name: "vibraphone", program: 11, family_mask: 16 },
    Instrument { name: "marimba", program: 12, family_mask: 16 },
    Instrument { name: "xylophone", program: 13, family_mask: 16 },
    Instrument { name: "tubular_bells", program: 14, family_mask: 16 },
    Instrument { name: "dulcimer", program: 15, family_mask: 144 },
    Instrument { name: "kalimba", program: 108, family_mask: 16 },
    Instrument { name: "tinkle_bell", program: 112, family_mask: 16 },
    Instrument { name: "steel_drums", program: 114, family_mask: 16 },
    Instrument { name: "organ", program: 16, family_mask: 32 },
    Instrument { name: "church_organ", program: 19, family_mask: 32 },
    Instrument { name: "reed_organ", program: 20, family_mask: 32 },
    Instrument { name: "accordion", program: 21, family_mask: 32 },
    Instrument { name: "harmonica", program: 22, family_mask: 288 },
    Instrument { name: "acoustic_guitar", program: 24, family_mask: 128 },
    Instrument { name: "electric_guitar", program: 26, family_mask: 128 },
    Instrument { name: "sitar", program: 104, family_mask: 128 },
    Instrument { name: "banjo", program: 105, family_mask: 128 },
    Instrument { name: "shamisen", program: 106, family_mask: 128 },
    Instrument { name: "koto", program: 107, family_mask: 128 },
    Instrument { name: "double_bass", program: 32, family_mask: 1025 },
    Instrument { name: "electric_bass", program: 33, family_mask: 1 },
    Instrument { name: "synth_bass", program: 38, family_mask: 2049 },
    Instrument { name: "violin", program: 40, family_mask: 1024 },
    Instrument { name: "viola", program: 41, family_mask: 1024 },
    Instrument { name: "cello", program: 42, family_mask: 1024 },
    Instrument { name: "string_ensemble", program: 44, family_mask: 1024 },
    Instrument { name: "synth_strings", program: 50, family_mask: 3072 },
    Instrument { name: "voice", program: 52, family_mask: 4096 },
    Instrument { name: "synth_voice", program: 54, family_mask: 6144 },
    Instrument { name: "trumpet", program: 56, family_mask: 2 },
    Instrument { name: "trombone", program: 57, family_mask: 2 },
    Instrument { name: "tuba", program: 58, family_mask: 2 },
    Instrument { name: "french_horn", program: 60, family_mask: 2 },
    Instrument { name: "brass_section", program: 61, family_mask: 2 },
    Instrument { name: "synth_brass", program: 62, family_mask: 2050 },
    Instrument { name: "soprano_sax", program: 64, family_mask: 256 },
    Instrument { name: "alto_sax", program: 65, family_mask: 256 },
    Instrument { name: "tenor_sax", program: 66, family_mask: 256 },
    Instrument { name: "baritone_sax", program: 67, family_mask: 256 },
    Instrument { name: "oboe", program: 68, family_mask: 256 },
    Instrument { name: "english_horn", program: 69, family_mask: 256 },
    Instrument { name: "bassoon", program: 70, family_mask: 256 },
    Instrument { name: "clarinet", program: 71, family_mask: 256 },
    Instrument { name: "bagpipe", program: 109, family_mask: 256 },
    Instrument { name: "shanai", program: 111, family_mask: 256 },
    Instrument { name: "piccolo", program: 72, family_mask: 4 },
    Instrument { name: "flute", program: 73, family_mask: 4 },
    Instrument { name: "recorder", program: 74, family_mask: 4 },
    Instrument { name: "pan_flute", program: 75, family_mask: 4 },
    Instrument { name: "blown_bottle", program: 76, family_mask: 4 },
    Instrument { name: "shakuhachi", program: 77, family_mask: 4 },
    Instrument { name: "whistle", program: 78, family_mask: 4 },
    Instrument { name: "ocarina", program: 79, family_mask: 4 },
    Instrument { name: "orchestra_hit", program: 55, family_mask: 2048 },
    Instrument { name: "synth_lead", program: 80, family_mask: 2048 },
    Instrument { name: "synth_pad", program: 88, family_mask: 2048 },
    Instrument { name: "synth_fx", program: 96, family_mask: 2048 },
    Instrument { name: "timpani", program: 47, family_mask: 64 },
    Instrument { name: "agogo", program: 113, family_mask: 64 },
    Instrument { name: "woodblock", program: 115, family_mask: 64 },
    Instrument { name: "taiko", program: 116, family_mask: 64 },
    Instrument { name: "melodic_tom", program: 117, family_mask: 64 },
    Instrument { name: "synth_drum", program: 118, family_mask: 2112 },
    Instrument { name: "reverse_cymbal", program: 119, family_mask: 64 },
    Instrument { name: "sound_effect", program: 120, family_mask: 512 },
];

pub fn credit(predicted: usize, actual: usize) -> f64 {
    if predicted == actual { return 1.0; }
    let a = INSTRUMENTS[predicted].family_mask;
    let b = INSTRUMENTS[actual].family_mask;
    if a & b != 0 { 0.5 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn instrument_vocabulary_and_credit_match_python() {
        assert_eq!(INSTRUMENTS.len(), 73);
        assert_eq!(INSTRUMENTS[0].name, "piano");
        assert_eq!(INSTRUMENTS[0].program, 0);
        assert_eq!(credit(0, 0), 1.0);
        assert_eq!(credit(0, 1), 0.5);
        assert_eq!(credit(0, 22), 0.0);
    }
}
