use anyhow::{Result, bail, ensure};
use std::collections::BTreeMap;
use serde::{Serialize, Deserialize};
use xml::reader::{EventReader, XmlEvent};

use crate::frontend::{CleanText, ModelConfig};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SileroControlConstants {
    pub point_id: i64,
    pub comma_ids: Vec<i64>,
    pub punctuation_ids: Vec<i64>,
    pub pitch_mean_shift: Vec<f32>,
    pub sil_value: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub text: String,
    pub clean_text: String,
    pub break_frames: Option<usize>,
    pub rate: f32,
    pub pitch: f32,
}

#[derive(Clone, Debug)]
struct Node {
    tag: String,
    attrs: BTreeMap<String, String>,
    text: String,
    children: Vec<Node>,
    tail: String,
}

#[derive(Clone, Debug)]
struct Parsed {
    text: String,
    break_strength: Option<String>,
    break_ms: Option<u32>,
    rate: f32,
    pitch: f32,
}

impl Parsed {
    fn new(text: String, rate: f32, pitch: f32) -> Self {
        Self { text, break_strength: None, break_ms: None, rate, pitch }
    }

    fn add_break(&mut self, strength: &str, duration_ms: u32) {
        self.break_strength = Some(strength.to_string());
        self.break_ms = Some(self.break_ms.unwrap_or(0).max(duration_ms));
    }
}

fn parse_document(xml: &str) -> Result<Node> {
    let mut root: Option<Node> = None;
    let mut stack: Vec<Node> = Vec::new();
    for event in EventReader::from_str(xml) {
        match event? {
            XmlEvent::StartElement { name, attributes, .. } => {
                stack.push(Node {
                    tag: match name.namespace { Some(namespace) => format!("{{{namespace}}}{}", name.local_name), None => name.local_name },
                    attrs: attributes.into_iter().map(|a| (a.name.local_name, a.value)).collect(),
                    text: String::new(), children: Vec::new(), tail: String::new(),
                });
            }
            XmlEvent::Characters(value) | XmlEvent::Whitespace(value) | XmlEvent::CData(value) => {
                if let Some(parent) = stack.last_mut() {
                    if let Some(last) = parent.children.last_mut() { last.tail.push_str(&value); }
                    else { parent.text.push_str(&value); }
                }
            }
            XmlEvent::EndElement { .. } => {
                let node = stack.pop().ok_or_else(|| anyhow::anyhow!("Unbalanced SSML element"))?;
                if let Some(parent) = stack.last_mut() { parent.children.push(node); }
                else if root.replace(node).is_some() { bail!("Multiple SSML roots"); }
            }
            _ => (),
        }
    }
    ensure!(stack.is_empty(), "Unclosed SSML element");
    root.ok_or_else(|| anyhow::anyhow!("SSML document is empty"))
}

fn strength_ms(name: &str) -> Option<u32> {
    match name {
        "x_weak" => Some(25), "weak" => Some(75), "medium" => Some(150),
        "strong" => Some(300), "x-strong" => Some(1000), _ => None,
    }
}

fn parse_break(attrs: &BTreeMap<String, String>) -> Result<(String, u32)> {
    let mut strength = attrs.get("strength").map(String::as_str).unwrap_or("medium");
    let duration = if let Some(value) = attrs.get("time") {
        let ms = if let Some(number) = value.strip_suffix("ms") {
            number.parse::<u32>()?
        } else if let Some(number) = value.strip_suffix('s') {
            number.parse::<u32>()?.saturating_mul(1000)
        } else {
            bail!("SSML break time must have an 'ms' or 's' suffix");
        };
        if ms >= 1000 { strength = "x-strong"; }
        else if ms >= 300 { strength = "strong"; }
        ms
    } else {
        strength_ms(strength).ok_or_else(|| anyhow::anyhow!("Invalid SSML break strength {strength:?}"))?
    };
    Ok((strength.to_string(), duration.min(5000)))
}

fn parse_prosody(attrs: &BTreeMap<String, String>) -> Result<(Option<f32>, Option<f32>)> {
    let rate = match attrs.get("rate").map(String::as_str) {
        None => None,
        Some(value) if value.ends_with('%') => Some(value.trim_end_matches('%').parse::<i32>()? as f32 / 100.0),
        Some("x-slow") => Some(0.5), Some("slow") => Some(0.8),
        Some("medium") => Some(1.0), Some("fast") => Some(1.2), Some("x-fast") => Some(1.5),
        Some(value) => bail!("Invalid SSML prosody rate {value:?}"),
    };
    let pitch = match attrs.get("pitch").map(String::as_str) {
        None => None,
        Some(value) if value.ends_with('%') => {
            let (sign, value) = value.split_at(1);
            let value = value.trim_end_matches('%').parse::<i32>()? as f32 / 100.0;
            match sign { "+" => Some(1.0 + value), "-" => Some(1.0 - value), _ => bail!("Invalid SSML pitch percentage") }
        },
        Some("x-low") => Some(0.75), Some("low") => Some(0.85),
        Some("medium") => Some(1.0), Some("high") => Some(1.15),
        Some("x-high") => Some(1.25), Some("robot") => Some(0.0),
        Some(value) => bail!("Invalid SSML prosody pitch {value:?}"),
    };
    ensure!(rate.is_some() || pitch.is_some(), "Empty SSML prosody tag");
    Ok((rate, pitch))
}

fn clean_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn push_text(out: &mut Vec<Parsed>, raw: &str, rate: f32, pitch: f32) {
    out.push(Parsed::new(clean_text(&raw.replace('\n', "")), rate, pitch));
}

fn visit(node: &Node, rate: f32, pitch: f32) -> Result<Vec<Parsed>> {
    let mut parsed = Vec::new();
    if !node.text.is_empty() { push_text(&mut parsed, &node.text, rate, pitch); }
    let mut last_tag: Option<&str> = None;
    for child in &node.children {
        match child.tag.as_str() {
            "break" => {
                let (strength, time) = parse_break(&child.attrs)?;
                if parsed.is_empty() { parsed.push(Parsed::new(".".into(), rate, pitch)); }
                parsed.last_mut().unwrap().break_strength = Some(strength);
                parsed.last_mut().unwrap().break_ms = Some(time);
            }
            "prosody" => {
                let (new_rate, new_pitch) = parse_prosody(&child.attrs)?;
                parsed.extend(visit(child, new_rate.unwrap_or(rate), new_pitch.unwrap_or(pitch))?);
            }
            "p" | "s" => {
                let strength = if child.tag == "s" { "strong" } else { "x-strong" };
                let mut child_parsed = visit(child, rate, pitch)?;
                if let Some(last) = parsed.last_mut() {
                    if !last.text.is_empty() || last_tag.is_some_and(|t| t != child.tag) || last_tag.is_none() {
                        last.add_break(strength, strength_ms(strength).unwrap());
                    }
                }
                if let Some(last) = child_parsed.last_mut() {
                    last.add_break(strength, strength_ms(strength).unwrap());
                    parsed.extend(child_parsed);
                }
            }
            _ => (),
        }
        last_tag = Some(child.tag.as_str());
        if !child.tail.is_empty() {
            let mut tail = child.tail.as_str();
            if let Some(first) = tail.chars().next() {
                if ".,!?…–;:".contains(first) {
                    if let Some(last) = parsed.last_mut() {
                        last.text = format!("{}{}", last.text.trim(), first);
                        tail = &tail[first.len_utf8()..];
                    }
                }
            }
            push_text(&mut parsed, tail, rate, pitch);
        }
    }
    Ok(parsed)
}

pub fn process_ssml(config: &ModelConfig, input: &str) -> Result<Vec<Segment>> {
    let document = clean_text(input);
    let root = parse_document(&document)?;
    ensure!(root.tag == "speak", "SSML requires a <speak> root");
    let mut result: Vec<Segment> = Vec::new();
    for (index, item) in visit(&root, 1.0, 1.0)?.into_iter().enumerate() {
        let CleanText { mut sentence, clean_sentence } = config.prepare_text_fragment(&item.text)?;
        let frames = item.break_ms.map(|ms| ((ms as f32) / 12.5) as usize);
        let has_text = clean_sentence.chars().any(|c| ('а'..='я').contains(&c));
        if has_text || index == 0 {
            if sentence.is_empty() || item.break_strength.is_some() && !sentence.chars().last().is_some_and(|c| "!,-.:;?–…".contains(c)) {
                sentence.push('.');
            }
            result.push(Segment { text: sentence, clean_text: clean_sentence, break_frames: frames, rate: item.rate, pitch: item.pitch });
        } else if item.break_strength.is_some() {
            if let Some(last) = result.last_mut() {
                if last.text.is_empty() || !last.text.chars().last().is_some_and(|c| "!,-.:;?–…".contains(c)) {
                    last.text.push('.');
                }
                last.break_frames = Some(last.break_frames.unwrap_or(0).max(frames.unwrap_or(0)));
            }
        }
    }
    ensure!(!result.is_empty() && result.iter().any(|s| !s.clean_text.is_empty()), "Text is empty");
    Ok(result)
}

#[derive(Clone, Debug, PartialEq)]
pub struct MergeInput {
    pub text: String,
    pub pauses: BTreeMap<usize, usize>,
    pub rates: Vec<f32>,
    pub pitches: Vec<f32>,
}

pub fn merge_segments(items: &[Segment], accented_full_text: &str) -> Result<MergeInput> {
    ensure!(!items.is_empty(), "Cannot merge empty SSML segments");
    let words = accented_full_text.split(' ').collect::<Vec<_>>();
    let mut cursor = 0;
    let mut accented_parts = Vec::new();
    for item in items {
        let count = item.text.split(' ').count();
        ensure!(cursor + count <= words.len(), "Accentor changed SSML word boundaries");
        accented_parts.push(words[cursor..cursor + count].join(" "));
        cursor += count;
    }
    ensure!(cursor == words.len(), "Accentor changed SSML word count");
    let mut pauses = BTreeMap::new();
    let mut rates = Vec::new();
    let mut pitches = Vec::new();
    let mut end = -1i64;
    for (index, (item, text)) in items.iter().zip(&accented_parts).enumerate() {
        let last = index + 1 == items.len();
        let count = text.chars().count() + if last { 2 } else { 1 };
        end += count as i64;
        if let Some(frames) = item.break_frames {
            let at = if last { end - 1 } else { end };
            ensure!(at >= 0, "SSML pause index is negative");
            pauses.insert(at as usize, frames);
        }
        rates.extend(std::iter::repeat_n(item.rate, count));
        pitches.extend(std::iter::repeat_n(item.pitch, count));
    }
    let text = accented_parts.join(" ");
    ensure!(rates.len() == text.chars().count() + 2, "SSML prosody vectors do not align with tokens");
    ensure!(rates.iter().all(|r| r.is_finite() && *r > 0.0), "SSML rate must be positive");
    ensure!(pitches.iter().all(|p| p.is_finite()), "Invalid SSML pitch");
    Ok(MergeInput { text, pauses, rates, pitches })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_ssml() {
        let config = test_config();
        assert!(process_ssml(&config, "<speak><prosody>тест</prosody></speak>").is_err());
        assert!(process_ssml(&config, "<speak>тест<break time=\"4min\"/></speak>").is_err());
        assert!(process_ssml(&config, "<not-speak>тест</not-speak>").is_err());
        assert!(process_ssml(&config, "<speak xmlns=\"urn:other\">тест</speak>").is_err());
        assert!(process_ssml(&config, "<speak><break/></speak>").is_err());
    }

    fn test_config() -> ModelConfig {
        ModelConfig {
            format_version: 1,
            symbols: "_~|!+,-.:;?абвгдежзийклмнопрстуфхцчшщъыьэюяё–… ".into(),
            alphabet: "абвгдежзийклмнопрстуфхцчшщъыьэюяё+".into(),
            sos: "|".into(), eos: "~".into(),
            speakers: [("xenia".to_string(),4)].into(),
            sample_rates: vec![8000,24000,48000],
        }
    }

    #[test]
    fn long_text_preparation_preserves_segment_indices() {
        let config = test_config();
        let input = "привет ".repeat(160);
        let ssml = format!("<speak>{input}<prosody rate=\"slow\">мир</prosody></speak>");
        let parts = process_ssml(&config, &ssml).unwrap();
        assert_eq!(parts.len(), 2);
        let accented = parts.iter().map(|p| p.text.as_str()).collect::<Vec<_>>().join(" ");
        let merged = merge_segments(&parts, &accented).unwrap();
        let (tokens, _) = config.encode_accented(&merged.text).unwrap();
        assert_eq!(tokens.len(), merged.rates.len());
        assert_eq!(tokens.len(), merged.pitches.len());
        assert_eq!(*merged.rates.last().unwrap(), 0.8);
    }

    #[test]
    fn matches_original_ssml_segments_and_merging() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = std::path::Path::new(&dir);
        let config: ModelConfig = serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let fixture: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(dir.join("ssml-parity.json")).unwrap()).unwrap();
        for case in fixture {
            let text = case["ssml"].as_str().unwrap();
            let parsed = process_ssml(&config, text).unwrap();
            let expected = case["segments"].as_array().unwrap();
            assert_eq!(parsed.len(), expected.len(), "Segment count differs: {text}");
            for (actual, expected) in parsed.iter().zip(expected) {
                assert_eq!(actual.text, expected["text"].as_str().unwrap(), "Text: {text}");
                assert_eq!(actual.clean_text, expected["clean_text"].as_str().unwrap(), "Clean text: {text}");
                assert_eq!(actual.break_frames, expected["break_time"].as_u64().map(|v| v as usize), "Break: {text}");
                assert!((actual.rate - expected["prosody_rate"].as_f64().unwrap() as f32).abs() < 1e-6, "Rate: {text}");
                assert!((actual.pitch - expected["prosody_pitch"].as_f64().unwrap() as f32).abs() < 1e-6, "Pitch: {text}");
            }
            let merged = merge_segments(&parsed, case["accented"].as_str().unwrap()).unwrap();
            let (ids, _) = config.encode_accented(&merged.text).unwrap();
            let expected_ids: Vec<i64> = serde_json::from_value(case["sequence"].clone()).unwrap();
            assert_eq!(ids, expected_ids, "Token IDs: {text}");
            let pauses: BTreeMap<usize, usize> = serde_json::from_value(case["pauses"].clone()).unwrap();
            assert_eq!(merged.pauses, pauses, "Pause positions: {text}");
            for (label, actual) in [("rates", merged.rates), ("pitches", merged.pitches)] {
                let expected: Vec<f32> = serde_json::from_value(case[label].clone()).unwrap();
                assert_eq!(actual.len(), expected.len(), "{label} length: {text}");
                for (a, e) in actual.iter().zip(&expected) {
                    assert!((a-e).abs() < 1e-6, "{label} mismatch: {text}");
                }
            }
        }
    }
}
