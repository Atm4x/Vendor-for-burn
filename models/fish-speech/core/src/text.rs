use crate::error::{FishError, Result};

pub fn clean_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.trim().chars() {
        let c = match c {
            '\u{2018}' | '\u{2019}' => '\'',
            _ => c,
        };
        if matches!(c as u32, 0x1F600..=0x1F64F | 0x1F300..=0x1F5FF | 0x1F680..=0x1F6FF | 0x1F1E0..=0x1F1FF)
        {
            continue;
        }
        if c == ',' && out.ends_with(',') {
            continue;
        }
        out.push(c);
    }
    out
}

fn protect_floats(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_numeric() {
            let mut dot = i;
            while dot < chars.len() && chars[dot].is_numeric() {
                dot += 1;
            }
            if dot < chars.len()
                && chars[dot] == '.'
                && dot + 1 < chars.len()
                && chars[dot + 1].is_numeric()
            {
                let mut end = dot + 1;
                while end < chars.len() && chars[end].is_numeric() {
                    end += 1;
                }
                out.push('<');
                for &c in &chars[i..dot] {
                    out.push(c);
                }
                out.push_str("_f_");
                for &c in &chars[dot + 1..end] {
                    out.push(c);
                }
                out.push('>');
                i = end;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn unprotect_floats(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_numeric() {
                j += 1;
            }
            if j > i + 1 && chars[j..].starts_with(&['_', 'f', '_']) {
                let k = j + 3;
                let mut end = k;
                while end < chars.len() && chars[end].is_numeric() {
                    end += 1;
                }
                if end > k && end < chars.len() && chars[end] == '>' {
                    for &c in &chars[i + 1..j] {
                        result.push(c);
                    }
                    result.push('.');
                    for &c in &chars[k..end] {
                        result.push(c);
                    }
                    i = end + 1;
                    continue;
                }
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

fn break_at(texts: Vec<String>, limit: usize, separators: &[char]) -> Vec<String> {
    let mut out = Vec::new();
    for text in texts {
        if text.len() <= limit {
            out.push(text);
            continue;
        }
        let mut current = String::new();
        for ch in text.chars() {
            current.push(ch);
            if separators.contains(&ch) {
                out.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
    out
}

fn break_at_bytes(texts: Vec<String>, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    for text in texts {
        if text.len() <= limit {
            out.push(text);
            continue;
        }
        let mut current = String::new();
        for ch in text.chars() {
            current.push(ch);
            if current.len() >= limit {
                out.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
    out
}

fn append_cleaned(current: &str, segments: &mut Vec<String>) {
    let value = current.trim();
    if !value.is_empty()
        && !value
            .chars()
            .all(|c| c.is_whitespace() || c.is_ascii_punctuation())
    {
        segments.push(value.to_string());
    }
}

pub fn split_text(text: &str, max_bytes: usize) -> Result<Vec<String>> {
    if max_bytes == 0 {
        return Err(FishError::InvalidConfig(
            "text segment byte limit must be positive",
        ));
    }
    let cleaned = clean_text(text);
    let mut parts = vec![protect_floats(&cleaned)];
    parts = break_at(parts, max_bytes, &['.', '!', '?', '。', '！', '？']);
    parts = parts.into_iter().map(|s| unprotect_floats(&s)).collect();
    parts = break_at(parts, max_bytes, &[',', '，']);
    parts = break_at(parts, max_bytes, &[' ']);
    parts = break_at_bytes(parts, max_bytes);
    let mut segments = Vec::new();
    let mut current = String::new();
    for part in parts {
        if current.len() + part.len() <= max_bytes {
            current.push_str(&part);
        } else {
            append_cleaned(&current, &mut segments);
            current = part;
        }
    }
    if !current.is_empty() {
        append_cleaned(&current, &mut segments);
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_cleaning_contract() {
        assert_eq!(
            clean_text("  ‘test’ 🤖🚀,,,,3.14...  "),
            "'test' 🤖,3.14..."
        );
        assert_eq!(clean_text("a,,,,b,,,c"), "a,b,c");
        assert_eq!(clean_text("\tHello\n"), "Hello");
    }

    #[test]
    fn python_split_examples() {
        assert_eq!(
            split_text(
                "This is a test sentence. This is another test sentence. And a third one.",
                50
            )
            .unwrap(),
            vec![
                "This is a test sentence.",
                "This is another test sentence. And a third one."
            ]
        );
        assert_eq!(
            split_text("a,aaaaaa3.14", 10).unwrap(),
            vec!["a,", "aaaaaa3.14"]
        );
        assert!(split_text("   ", 10).unwrap().is_empty());
        assert_eq!(split_text("a", 10).unwrap(), vec!["a"]);
        assert!(split_text("a", 0).is_err());
    }

    #[test]
    fn chinese_multibyte_and_float() {
        let text = "这是一段很长的中文文本,而且没有句号,也没有感叹号,也没有问号,也没有换行符。";
        assert_eq!(
            split_text(text, 50).unwrap(),
            vec![
                "这是一段很长的中文文本,",
                "而且没有句号,也没有感叹号,",
                "也没有问号,也没有换行符。"
            ]
        );
        assert_eq!(
            split_text("3.14. 15.92. check", 10).unwrap(),
            vec!["3.14.", "15.92.", "check"]
        );
    }
}

#[cfg(test)]
mod original_python_fixture {
    use super::split_text;

    fn decode_hex(value: &str) -> String {
        assert_eq!(value.len() % 2, 0);
        let bytes = value
            .as_bytes()
            .chunks_exact(2)
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        String::from_utf8(bytes).unwrap()
    }

    #[test]
    fn original_python_corpus_matches() {
        let data = include_str!("../fixtures/text_split_original_python.tsv");
        let mut total = 0;
        for line in data.lines() {
            let cols = line.split('\t').collect::<Vec<_>>();
            assert_eq!(cols.len(), 3);
            let size = cols[0].parse::<usize>().unwrap();
            let input = decode_hex(cols[1]);
            let expected = if cols[2].is_empty() {
                Vec::new()
            } else {
                cols[2].split('|').map(decode_hex).collect()
            };
            let actual = split_text(&input, size).unwrap();
            assert_eq!(
                actual, expected,
                "original Python mismatch at case {total}: {input:?}, max_bytes={size}"
            );
            total += 1;
        }
        assert_eq!(total, 144);
    }
}
