use crate::{F5Error, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SilenceRemoval {
    pub min_silence_ms: usize,
    pub threshold_dbfs: f64,
    pub keep_silence_ms: usize,
    pub seek_step_ms: usize,
}

impl Default for SilenceRemoval {
    fn default() -> Self {
        Self {
            min_silence_ms: 1000,
            threshold_dbfs: -50.0,
            keep_silence_ms: 500,
            seek_step_ms: 10,
        }
    }
}

impl SilenceRemoval {
    pub fn validate(self) -> Result<()> {
        if self.min_silence_ms == 0 || self.seek_step_ms == 0
            || !self.threshold_dbfs.is_finite()
            || !(-120.0..=0.0).contains(&self.threshold_dbfs)
        {
            return Err(F5Error::InvalidConfiguration("Invalid silence-removal settings"));
        }
        Ok(())
    }
}

/// Reproduces the original generated-WAV pydub split_on_silence for mono PCM16.
pub fn remove_generated_silence(
    input: &[i16],
    sample_rate: u32,
    options: SilenceRemoval,
) -> Result<Vec<i16>> {
    options.validate()?;
    if sample_rate == 0 {
        return Err(F5Error::InvalidConfiguration("Silence removal requires sample rate"));
    }
    let rate = sample_rate as usize;
    let length = (input.len() as f64 * 1000.0 / rate as f64).round() as usize;
    let minimum = options.min_silence_ms;
    if length < minimum {
        return Ok(input.to_vec());
    }
    let to_frame = |ms: usize| ms.saturating_mul(rate) / 1000;
    let mut prefix = Vec::with_capacity(input.len() + 1);
    prefix.push(0u64);
    for &sample in input {
        let value = i64::from(sample);
        prefix.push(prefix.last().copied().unwrap_or(0).saturating_add((value * value) as u64));
    }
    let threshold = 32768.0 * 10_f64.powf(options.threshold_dbfs / 20.0);
    let last = length - minimum;
    let mut silence = Vec::new();
    let is_silent = |ms: usize| -> bool {
        let start = to_frame(ms).min(input.len());
        let end = to_frame(ms + minimum).min(input.len());
        end > start && ((prefix[end] - prefix[start]) as f64 / (end - start) as f64)
            .sqrt().floor() <= threshold
    };
    let mut ms = 0;
    while ms <= last {
        if is_silent(ms) {
            silence.push(ms);
        }
        ms = match ms.checked_add(options.seek_step_ms) {
            Some(next) => next,
            None => break,
        };
    }
    if last % options.seek_step_ms != 0 && is_silent(last) {
        silence.push(last);
    }
    if silence.is_empty() {
        return Ok(input.to_vec());
    }
    let mut silent_ranges = Vec::new();
    let mut range_start = silence[0];
    let mut previous = silence[0];
    for &current in &silence[1..] {
        if current != previous + options.seek_step_ms && current > previous + minimum {
            silent_ranges.push((range_start, previous + minimum));
            range_start = current;
        }
        previous = current;
    }
    silent_ranges.push((range_start, previous + minimum));
    let mut nonsilent = Vec::new();
    let mut previous_end = 0;
    for (start, end) in silent_ranges {
        if previous_end < start {
            nonsilent.push((previous_end, start));
        }
        previous_end = end;
    }
    if previous_end < length {
        nonsilent.push((previous_end, length));
    }
    if nonsilent.is_empty() {
        return Ok(Vec::new());
    }
    let mut kept: Vec<(usize, usize)> = nonsilent.into_iter()
        .map(|(start, end)| (
            start.saturating_sub(options.keep_silence_ms),
            (end + options.keep_silence_ms).min(length),
        ))
        .collect();
    for index in 1..kept.len() {
        if kept[index].0 < kept[index - 1].1 {
            let middle = (kept[index - 1].1 + kept[index].0) / 2;
            kept[index - 1].1 = middle;
            kept[index].0 = middle;
        }
    }
    let mut output = Vec::new();
    for (start, end) in kept {
        let a = to_frame(start).min(input.len());
        let b = to_frame(end).min(input.len());
        output.extend_from_slice(&input[a..b]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pydub_generated_wav_silence_patterns_match_exact_pcm16() {
        let cases: [&[(i16, usize)]; 6] = [
            &[(10000,100),(0,36000),(10000,36000),(0,54000),(-9000,24000)],
            &[(0,48000),(7000,24000),(0,72000),(7000,24000),(0,48000)],
            &[(6000,48000),(0,24000),(6000,48000)],
            &[(0,72000)],
            &[(5000,24000)],
            &[(4000,12000),(0,60000),(4000,12000),(0,48000),(4000,12000)],
        ];
        let expected: [&[(i16, usize)]; 6] = [
            &[(10000,100),(0,24240),(10000,36000),(0,24240),(-9000,23996)],
            &[(0,12000),(7000,24000),(0,24000),(7000,24000),(0,12000)],
            &[(6000,48000),(0,24000),(6000,48000)],
            &[],
            &[(5000,24000)],
            &[(4000,12000),(0,24000),(4000,12000),(0,24000),(4000,12000)],
        ];
        for (index, input) in cases.iter().enumerate() {
            let values: Vec<i16> = input.iter()
                .flat_map(|&(value, len)| std::iter::repeat_n(value, len)).collect();
            let result = remove_generated_silence(&values, 24_000, SilenceRemoval::default()).unwrap();
            let oracle: Vec<i16> = expected[index].iter()
                .flat_map(|&(value, len)| std::iter::repeat_n(value, len)).collect();
            assert_eq!(result, oracle, "pydub case {index}");
        }
    }

    #[test]
    fn invalid_config_is_rejected() {
        let mut config = SilenceRemoval::default();
        config.seek_step_ms = 0;
        assert!(remove_generated_silence(&[], 24_000, config).is_err());
    }
}
