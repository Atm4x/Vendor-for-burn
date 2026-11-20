use super::repair::RepairInput;
use std::f64::consts::PI;

pub const CONTINUOUS_FEATURES: usize = 110;

fn fourier(x: f64, n: usize, low: f64, high: f64, out: &mut Vec<f32>) {
    let theta = (0..n).map(|k| {
        let frac = k as f64 / (n - 1) as f64;
        let wavelength = (low.ln() + frac * (high.ln() - low.ln())).exp();
        2.0 * PI * x / wavelength
    }).collect::<Vec<_>>();
    out.extend(theta.iter().map(|t| t.sin() as f32));
    out.extend(theta.iter().map(|t| t.cos() as f32));
}

fn signed_log(x: f64) -> f32 { (x.signum() * (1.0 + x.abs() * 20.0).ln()) as f32 }

pub fn context_features(notes: &[RepairInput], index: usize) -> [f32; 8] {
    let n = &notes[index];
    let mut prev_time = 10.0f64;
    let mut prev_off = 0.0;
    let mut next_time = 10.0f64;
    let mut chord = 0usize;
    let mut dense = 0usize;
    let mut near_key = 0usize;
    for (i, other) in notes.iter().enumerate() {
        let dt = other.onset - n.onset;
        if dt.abs() < 0.035 { chord += 1; }
        if dt.abs() < 0.5 { dense += 1; }
        if i == index || other.pitch != n.pitch { continue; }
        if dt.abs() < 0.25 { near_key += 1; }
        if dt <= 0.0 && -dt < prev_time { prev_time = -dt; prev_off = other.offset; }
        if dt > 0.0 && dt < next_time { next_time = dt; }
    }
    let prev_gap_off = if prev_time < 10.0 { n.onset - prev_off } else { 10.0 };
    let next_overlap = if next_time < 10.0 { n.offset - n.onset - next_time } else { -10.0 };
    [
        signed_log(prev_time), signed_log(prev_gap_off),
        signed_log(next_time), signed_log(next_overlap),
        (1.0 + chord as f64).ln() as f32,
        (1.0 + dense as f64).ln() as f32,
        (1.0 + dense as f64).ln() as f32,
        (1.0 + near_key as f64).ln() as f32,
    ]
}

pub fn grid_features(notes: &[RepairInput]) -> Vec<[f32; 28]> {
    if notes.is_empty() { return Vec::new(); }
    let periods = (0..72).map(|i| {
        (0.18f64.ln() + i as f64 * (1.5f64.ln() - 0.18f64.ln()) / 71.0).exp()
    }).collect::<Vec<_>>();
    let n = notes.len() as f64;
    let mut strengths = vec![0.0f64; 72];
    let mut phases = vec![0.0f64; 72];
    for (j, &period) in periods.iter().enumerate() {
        let mut re = 0.0;
        let mut im = 0.0;
        for note in notes {
            let theta = 2.0 * PI * note.onset / period;
            re += theta.cos();
            im += theta.sin();
        }
        strengths[j] = re.hypot(im) / n;
        phases[j] = im.atan2(re);
    }
    let mut available = strengths.clone();
    let mut out = vec![[0.0; 28]; notes.len()];
    for grid in 0..2 {
        let best = (0..72).max_by(|&a, &b| {
            available[a].total_cmp(&available[b]).then_with(|| b.cmp(&a))
        }).unwrap();
        let period = periods[best];
        for (row, note) in out.iter_mut().zip(notes) {
            let base = 2.0 * PI * note.onset / period - phases[best];
            for (k, harmonic) in [1, 2, 3, 4, 6, 8].iter().enumerate() {
                row[grid * 14 + k * 2] = (f64::from(*harmonic) * base).sin() as f32;
                row[grid * 14 + k * 2 + 1] = (f64::from(*harmonic) * base).cos() as f32;
            }
            row[grid * 14 + 12] = strengths[best] as f32;
            row[grid * 14 + 13] = period.ln() as f32;
        }
        for (i, v) in available.iter_mut().enumerate() {
            if i.abs_diff(best) <= 3 { *v = -1.0; }
        }
    }
    out
}

pub fn continuous_features(notes: &[RepairInput], window_start: f64) -> Result<Vec<[f32; CONTINUOUS_FEATURES]>, String> {
    if notes.iter().any(|n| !n.onset.is_finite() || !n.offset.is_finite()
        || n.offset < n.onset || n.pitch > 127)
    { return Err("invalid note for Evidence embedding".into()); }
    let local = notes.iter().map(|n| RepairInput {
        onset: n.onset - window_start,
        offset: n.offset - window_start,
        ..*n
    }).collect::<Vec<_>>();
    let grid = grid_features(&local);
    let mut output = Vec::with_capacity(local.len());
    for (i, note) in local.iter().enumerate() {
        let mut c = Vec::with_capacity(CONTINUOUS_FEATURES);
        let dur = (note.offset - note.onset).max(0.0001);
        let log_dur = dur.ln();
        fourier(note.onset, 12, 0.01, 32.0, &mut c);
        fourier(log_dur, 12, 0.04, 40.0, &mut c);
        fourier(note.offset, 12, 0.01, 32.0, &mut c);
        c.push(((log_dur + 3.0) / 3.0) as f32);
        c.push((dur.min(8.0) / 2.0) as f32);
        c.extend_from_slice(&context_features(&local, i));
        c.extend_from_slice(&grid[i]);
        output.push(c.try_into().map_err(|_| "invalid continuous embedding size")?);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn produces_evidence_context_and_tempo_features() {
        let notes = (0..5).map(|i| RepairInput {
            onset: i as f64 * 0.2, offset: i as f64 * 0.2 + 0.15,
            pitch: 60, velocity: 90, original_index: i,
        }).collect::<Vec<_>>();
        let rows = continuous_features(&notes, 0.0).unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].len(), 110);
        assert!(rows.iter().flatten().all(|v| v.is_finite()));
        assert_eq!(context_features(&notes, 0)[7], (1.0f32 + 1.0).ln());
    }
}
