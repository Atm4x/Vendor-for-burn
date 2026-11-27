use regex::Regex;
use std::sync::LazyLock;

static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)[а-яёa-z]+").unwrap());
static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)(?:[,\s]+(правда|верно|да)\s*\?$)|(?:[,\s]+(не\s+так\s+ли|не\s+правда\s+ли|разве\s+не\s+так|ведь\s+так)\s*\?$)|(?:[,\s]+ведь\s*\?$)|(?:[,\s]+а\s*\?$)").unwrap());
static ALTERNATIVE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\bили\b").unwrap());
const FILLERS: &str = "так слушай короче слышь и типа вобще скажите пожалуйста вообще скажи вот ну а";
const WH: &str = "какой чем каком чей когда которым отчего откуда скольких чьего которое сколько которых чьи где который какого каких какими каковы чьему чьим какому какое чьих что кто которому каким зачем какие куда почему чьём котором которого чо почём чему чём чё чья каково как которой насколько чьё которые которая которыми каков чью чьими чьей какую кем сколькими чьем чего которую ком кому какая чье какова скольким кого";

fn classify(text: &str) -> i64 {
    let mut s = text.trim();
    while s.starts_with(['«','"','„','“']) { s = s[s.chars().next().unwrap().len_utf8()..].trim(); }
    while s.ends_with(['’','"','”','»']) { s = s[..s.len()-s.chars().last().unwrap().len_utf8()].trim(); }
    let tail = if s.ends_with("?!") || s.ends_with("?..") || s.ends_with("?...") { s.trim_end_matches(['.','!']) } else { s };
    if tail.ends_with('?') {
        let cleaned = tail.replace('+', "").replace('ё', "е").replace('Ё', "Е");
        let normalized = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
        if TAG.is_match(&normalized) { return 4; }
        let lower = normalized.to_lowercase();
        let wh = WORD.find_iter(&lower).take(8).map(|m| m.as_str()).filter(|w| !FILLERS.split(' ').any(|x| x == *w)).take(4).any(|w| WH.split(' ').any(|x| x == w));
        if wh { return 1; }
        if ALTERNATIVE.is_match(&normalized) { return 3; }
        2
    } else if s.ends_with('!') { 5 } else { 0 }
}

fn sentence_ranges(chars: &[char]) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut i = 1;
    while i < chars.len() {
        if matches!(chars[i-1], '.'|'!'|'?') && chars[i].is_whitespace() {
            while i < chars.len() && chars[i].is_whitespace() { i += 1; }
            ranges.push(start..i);
            start = i;
        } else { i += 1; }
    }
    if start < chars.len() { ranges.push(start..chars.len()); }
    ranges
}

pub(crate) fn type_ids(raw: &str, accented: &str) -> Vec<i64> {
    let raw: Vec<_> = raw.trim().chars().collect();
    let tokens: Vec<_> = accented.chars().collect();
    let spans = sentence_ranges(&tokens);
    let mut types: Vec<_> = sentence_ranges(&raw).into_iter().map(|s| classify(&raw[s].iter().collect::<String>())).collect();
    if types.len() != spans.len() {
        types = spans.iter().map(|s| classify(&tokens[s.clone()].iter().collect::<String>().replace('+', ""))).collect();
    }
    let mut ids = Vec::with_capacity(tokens.len()+2);
    ids.push(types.first().copied().unwrap_or(0));
    for (i,span) in spans.iter().enumerate() { ids.extend(std::iter::repeat_n(types[i], span.len())); }
    ids.push(types.last().copied().unwrap_or(0));
    ids
}

pub(crate) fn legacy_type_ids(raw: &str, len: usize) -> Vec<i64> {
    let raw: Vec<_> = raw.trim().chars().collect();
    let spans = sentence_ranges(&raw);
    let first = spans.first().map(|s| classify(&raw[s.clone()].iter().collect::<String>())).unwrap_or(0);
    let mut ids = Vec::with_capacity(len);
    ids.push(first);
    for (i, span) in spans.iter().enumerate() {
        let text = raw[span.clone()].iter().collect::<String>();
        let count = text.trim_end_matches(char::is_whitespace).chars().count() + usize::from(i+1 < spans.len());
        ids.extend(std::iter::repeat_n(classify(&text), count));
    }
    ids.resize(len, first);
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn base_sentence_types_and_token_boundaries() {
        for (s,expected) in [("Я дома.",0),("Куда идёшь?",1),("Ты дома?",2),("Чай или кофе?",3),("Ты дома, да?",4),("Привет!",5),("«Ты дома?!»",2)] { assert_eq!(classify(s),expected,"{s}"); }
        assert_eq!(type_ids("Да. Нет!", "д+а. н+ет!"), vec![0,0,0,0,0,0,5,5,5,5,5,5]);
    }
}
