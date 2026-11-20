use super::{cluster::{average_linkage, cosine, maximum_matching, normalize, normalized64}, instruments::{INSTRUMENTS, credit}};
use std::collections::BTreeMap;

pub const INSTRUMENT_CLASSES: usize = 73;

#[derive(Clone, Debug)]
pub struct WindowPrediction {
    pub start_seconds: f64,
    pub note_indices: Vec<usize>,
    pub embeddings: Vec<Vec<f32>>,
    pub probabilities: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct GroupPredictions {
    pub window_seconds: f64,
    pub embeddings: Vec<Vec<f32>>,
    pub probabilities: Vec<Vec<f32>>,
    pub hidden: Option<Vec<Vec<f32>>>,
    pub windows: Vec<WindowPrediction>,
}

#[derive(Clone, Copy, Debug)]
pub struct GroupConfig {
    pub tau: f64,
    pub tau2: f64,
    pub lam: f64,
    pub min_notes: usize,
    pub refine: usize,
    pub alpha: f64,
    pub expected_credit: bool,
    pub merge_same_instrument: bool,
    pub absorb: f64,
    pub absorb_fraction: f64,
}

impl Default for GroupConfig {
    fn default() -> Self {
        Self {
            tau: 0.1, tau2: 0.4, lam: 0.5, min_notes: 30,
            refine: 2, alpha: 0.5, expected_credit: true,
            merge_same_instrument: true, absorb: 0.3, absorb_fraction: 0.25,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstrumentGroup {
    pub instrument_index: usize,
    pub tracklet_index: usize,
    pub note_indices: Vec<usize>,
}

pub fn validate_predictions(input: &GroupPredictions, onsets: &[f64]) -> Result<(), String> {
    let n = onsets.len();
    if input.embeddings.len() != n || input.probabilities.len() != n
        || !input.window_seconds.is_finite() || input.window_seconds <= 0.0
        || onsets.iter().any(|x| !x.is_finite())
    { return Err("invalid Grouper song dimensions".into()); }
    let emb = input.embeddings.first().map(Vec::len).unwrap_or(1);
    if emb == 0 || input.embeddings.iter().any(|x| x.len() != emb || x.iter().any(|v| !v.is_finite()))
        || input.probabilities.iter().any(|x| x.len() != INSTRUMENT_CLASSES
            || x.iter().any(|v| !v.is_finite() || *v < 0.0))
    { return Err("invalid Grouper note embeddings/probabilities".into()); }
    if let Some(hidden) = &input.hidden {
        if hidden.len() != n || hidden.iter().any(|row| row.iter().any(|x| !x.is_finite())) {
            return Err("invalid Grouper track-head vectors".into());
        }
    }
    for window in &input.windows {
        if !window.start_seconds.is_finite() || window.note_indices.len() != window.embeddings.len()
            || window.note_indices.len() != window.probabilities.len()
            || window.note_indices.iter().any(|&i| i >= n)
            || window.embeddings.iter().any(|e| e.len() != emb || e.iter().any(|v| !v.is_finite()))
            || window.probabilities.iter().any(|p| p.len() != INSTRUMENT_CLASSES
                || p.iter().any(|v| !v.is_finite() || *v < 0.0))
        { return Err("invalid Grouper window predictions".into()); }
    }
    Ok(())
}

pub fn group_chain(input: &GroupPredictions, onsets: &[f64], tau: f64, lam: f64)
    -> Result<Vec<Option<usize>>, String>
{
    validate_predictions(input, onsets)?;
    let n = onsets.len();
    let mut assigned = vec![None; n];
    let mut owner = vec![None; n];
    let mut best = vec![-1.0f64; n];
    let mut next_global = 0usize;
    let mut order = (0..input.windows.len()).collect::<Vec<_>>();
    order.sort_by(|&i, &j| input.windows[i].start_seconds.total_cmp(&input.windows[j].start_seconds));
    for wi in order {
        let window = &input.windows[wi];
        let m = window.note_indices.len();
        if m == 0 { continue; }
        let normalized = window.probabilities.iter().map(|p| normalize(p)).collect::<Vec<_>>();
        let mut distances = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in i + 1..m {
                let e = window.embeddings[i].iter().zip(&window.embeddings[j])
                    .map(|(&a, &b)| f64::from(a) * f64::from(b)).sum::<f64>();
                let distance = (1.0 - e + lam * (1.0 - cosine(&normalized[i], &normalized[j]))).max(0.0);
                distances[i][j] = distance;
                distances[j][i] = distance;
            }
        }
        let local = average_linkage(&distances, tau)?;
        let count = local.iter().copied().max().unwrap_or(0) + 1;
        let mut connected = vec![None; count];
        let previous = window.note_indices.iter().filter_map(|&j| owner[j]).collect::<Vec<_>>();
        let mut previous = previous;
        previous.sort_unstable();
        previous.dedup();
        if !previous.is_empty() {
            let mut overlap = vec![vec![0usize; previous.len()]; count];
            for (&note, &label) in window.note_indices.iter().zip(&local) {
                if let Some(old) = owner[note] {
                    if let Ok(col) = previous.binary_search(&old) { overlap[label][col] += 1; }
                }
            }
            for (row, col) in maximum_matching(&overlap)? {
                if overlap[row][col] > 0 { connected[row] = Some(previous[col]); }
            }
        }
        for cluster in &mut connected {
            if cluster.is_none() {
                *cluster = Some(next_global);
                next_global += 1;
            }
        }
        for (i, &note) in window.note_indices.iter().enumerate() {
            let id = connected[local[i]];
            owner[note] = id;
            let normalized_position = (onsets[note] - window.start_seconds) / input.window_seconds;
            let weight = (1.0 - (normalized_position - 0.5).abs() * 2.0).clamp(0.05, 1.0);
            if weight > best[note] {
                assigned[note] = id;
                best[note] = weight;
            }
        }
    }
    Ok(assigned)
}

pub fn group_tracklets(input: &GroupPredictions, onsets: &[f64], config: GroupConfig)
    -> Result<Vec<Option<usize>>, String>
{
    let assigned = group_chain(input, onsets, config.tau, config.lam)?;
    let mut tids = assigned.iter().flatten().copied().collect::<Vec<_>>();
    tids.sort_unstable(); tids.dedup();
    let m = tids.len();
    if m <= 1 { return Ok(assigned); }
    let edim = input.embeddings.first().map(Vec::len).unwrap_or(1);
    let mut sizes = vec![0usize; m];
    let mut me = vec![vec![0.0f64; edim]; m];
    let mut mp = vec![vec![0.0f64; INSTRUMENT_CLASSES]; m];
    for (note, id) in assigned.iter().enumerate() {
        let Some(id) = id else { continue; };
        let k = tids.binary_search(id).map_err(|_| "invalid tracklet id")?;
        sizes[k] += 1;
        for (dst, &src) in me[k].iter_mut().zip(&input.embeddings[note]) { *dst += f64::from(src); }
        let prob = normalize(&input.probabilities[note]);
        for (dst, src) in mp[k].iter_mut().zip(prob) { *dst += src; }
    }
    for k in 0..m {
        for v in &mut me[k] { *v /= sizes[k] as f64; }
        for v in &mut mp[k] { *v /= sizes[k] as f64; }
    }
    let mut alive = vec![true; m];
    let mut parents = (0..m).collect::<Vec<_>>();
    let mut blocked = vec![false; m];
    if m > 600 {
        let mut ranked = (0..m).collect::<Vec<_>>();
        ranked.sort_by_key(|&id| std::cmp::Reverse(sizes[id]));
        for &i in ranked.iter().skip(600) { blocked[i] = true; }
    }
    loop {
        let mut best: Option<(usize, usize, f64)> = None;
        for i in 0..m {
            if !alive[i] || blocked[i] { continue; }
            for j in i + 1..m {
                if !alive[j] || blocked[j] { continue; }
                let d = (1.0 - cosine(&me[i], &me[j]))
                    + config.lam * (1.0 - cosine(&mp[i], &mp[j]));
                if best.map(|(_, _, score)| d < score).unwrap_or(true) {
                    best = Some((i, j, d));
                }
            }
        }
        let Some((i, j, distance)) = best else { break; };
        if distance > config.tau2 { break; }
        let (si, sj) = (sizes[i] as f64, sizes[j] as f64);
        for channel in 0..edim { me[i][channel] = (si * me[i][channel] + sj * me[j][channel]) / (si + sj); }
        for channel in 0..INSTRUMENT_CLASSES { mp[i][channel] = (si * mp[i][channel] + sj * mp[j][channel]) / (si + sj); }
        sizes[i] += sizes[j];
        alive[j] = false;
        for id in &mut parents { if *id == j { *id = i; } }
    }
    Ok(assigned.into_iter().map(|id| id.map(|id| parents[tids.binary_search(&id).unwrap()])).collect())
}

fn centers(ids: &[usize], vectors: &[Vec<f32>], labels: &[usize], k: usize) -> Vec<Vec<f64>> {
    let d = vectors.first().map(Vec::len).unwrap_or(0);
    let mut out = vec![vec![0.0; d]; k];
    for (&id, &cluster) in ids.iter().zip(labels) {
        for (sum, &v) in out[cluster].iter_mut().zip(&vectors[id]) { *sum += f64::from(v); }
    }
    out
}

fn absorb_small(
    ids: &[usize], labels: &mut [usize], input: &GroupPredictions, fraction: f64, tau: f64,
) {
    if ids.is_empty() || fraction <= 0.0 || tau <= 0.0 { return; }
    let embedding = ids.iter().map(|&id| normalize(&input.embeddings[id])).collect::<Vec<_>>();
    loop {
        let mut keys = labels.to_vec(); keys.sort_unstable(); keys.dedup();
        if keys.len() < 2 { return; }
        let mut sizes = vec![0usize; keys.len()];
        let mut sums = vec![vec![0.0; embedding[0].len()]; keys.len()];
        for (i, &label) in labels.iter().enumerate() {
            let k = keys.binary_search(&label).unwrap();
            sizes[k] += 1;
            for (dst, &v) in sums[k].iter_mut().zip(&embedding[i]) { *dst += v; }
        }
        for (k, sum) in sums.iter_mut().enumerate() {
            for x in sum { *x /= sizes[k] as f64; }
        }
        let mut ranked = (0..keys.len()).collect::<Vec<_>>();
        ranked.sort_by_key(|&k| sizes[k]);
        let mut dissolved = false;
        for small in ranked {
            let share = sizes[small] as f64 / labels.len() as f64;
            if share >= fraction { break; }
            let bigger = (0..keys.len()).filter(|&k| sizes[k] > sizes[small]).collect::<Vec<_>>();
            if bigger.is_empty() { continue; }
            let closest = bigger.iter().map(|&k| 1.0 - cosine(&sums[small], &sums[k]))
                .fold(f64::INFINITY, f64::min);
            if closest >= tau * (1.0 - share / fraction) { continue; }
            for (i, label) in labels.iter_mut().enumerate() {
                if *label != keys[small] { continue; }
                let target = bigger.iter().copied().max_by(|&a, &b| {
                    let da = cosine(&embedding[i], &normalized64(&sums[a]));
                    let db = cosine(&embedding[i], &normalized64(&sums[b]));
                    da.total_cmp(&db).then_with(|| b.cmp(&a))
                }).unwrap();
                *label = keys[target];
            }
            dissolved = true;
            break;
        }
        if !dissolved { break; }
    }
}

pub fn decode_groups(
    input: &GroupPredictions, notes: &[(f64, f64, u8)], config: GroupConfig,
    track_head: Option<&dyn Fn(&[usize]) -> Result<Vec<f64>, String>>,
) -> Result<Vec<InstrumentGroup>, String> {
    let onsets = notes.iter().map(|n| n.0).collect::<Vec<_>>();
    validate_predictions(input, &onsets)?;
    if notes.iter().any(|(on, off, p)| !on.is_finite() || !off.is_finite() || off <= on || *p > 127) {
        return Err("invalid Grouper note times or pitches".into());
    }
    if !config.tau.is_finite() || !config.tau2.is_finite() || !config.lam.is_finite()
        || !config.absorb.is_finite() || !config.absorb_fraction.is_finite()
        || !config.alpha.is_finite() || !(0.0..=1.0).contains(&config.alpha)
    { return Err("invalid Grouper decoding parameters".into()); }
    if notes.is_empty() { return Ok(Vec::new()); }
    let tracks = group_tracklets(input, &onsets, config)?;
    let ids = (0..notes.len()).filter(|&i| tracks[i].is_some()).collect::<Vec<_>>();
    if ids.is_empty() { return Ok(Vec::new()); }
    let mut labels = ids.iter().map(|&i| tracks[i].unwrap()).collect::<Vec<_>>();
    let k = labels.iter().copied().max().unwrap() + 1;
    let mut size = vec![0usize; k];
    for &id in &labels { size[id] += 1; }
    for _ in 0..config.refine {
        let big = (0..k).filter(|&t| size[t] >= config.min_notes).collect::<Vec<_>>();
        if big.len() < 2 { break; }
        let e = centers(&ids, &input.embeddings, &labels, k);
        let p = centers(&ids, &input.probabilities, &labels, k);
        let ce = big.iter().map(|&t| normalized64(&e[t])).collect::<Vec<_>>();
        let cp = big.iter().map(|&t| normalized64(&p[t])).collect::<Vec<_>>();
        for (j, &id) in ids.iter().enumerate() {
            let prob = normalize(&input.probabilities[id]);
            let best = (0..big.len()).max_by(|&a, &b| {
                let sc = |slot: usize| {
                    let em = input.embeddings[id].iter().zip(&ce[slot])
                        .map(|(&x, &y)| f64::from(x) * y).sum::<f64>();
                    em + config.lam * cosine(&prob, cp[slot].as_slice())
                };
                sc(a).total_cmp(&sc(b)).then_with(|| b.cmp(&a))
            }).unwrap();
            labels[j] = big[best];
        }
        size.fill(0);
        for &id in &labels { size[id] += 1; }
    }
    let mut mass = centers(&ids, &input.probabilities, &labels, k);
    let big = {
        let chosen = (0..k).filter(|&t| size[t] >= config.min_notes).collect::<Vec<_>>();
        if chosen.is_empty() {
            vec![(0..k).max_by_key(|&t| size[t]).unwrap()]
        } else { chosen }
    };
    for p in &mut mass { *p = normalized64(p); }
    for label in &mut labels {
        if size[*label] == 0 || big.contains(label) { continue; }
        let selected = big.iter().copied().max_by(|&a, &b| {
            cosine(&mass[a], &mass[*label]).total_cmp(&cosine(&mass[b], &mass[*label]))
                .then_with(|| b.cmp(&a))
        }).unwrap();
        *label = selected;
    }
    absorb_small(&ids, &mut labels, input, config.absorb_fraction, config.absorb);
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for (&idx, &track) in ids.iter().zip(&labels) { groups.entry(track).or_default().push(idx); }
    let mut out = Vec::<InstrumentGroup>::new();
    let mut merged = BTreeMap::<usize, usize>::new();
    for (tracklet, indices) in groups {
        let mut probs = vec![0.0f64; INSTRUMENT_CLASSES];
        for &id in &indices {
            for (sum, &prob) in probs.iter_mut().zip(&input.probabilities[id]) { *sum += f64::from(prob); }
        }
        let total = probs.iter().sum::<f64>().max(1e-9);
        for prob in &mut probs { *prob = (*prob / total).max(1e-9).ln(); }
        if let Some(predictor) = track_head {
            if config.alpha > 0.0 {
                let head_logits = predictor(&indices)?;
                if head_logits.len() != INSTRUMENT_CLASSES || head_logits.iter().any(|x| !x.is_finite()) {
                    return Err("invalid Grouper track-head logits".into());
                }
                let max = head_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let z = head_logits.iter().map(|v| (v - max).exp()).sum::<f64>().ln() + max;
                for (p, h) in probs.iter_mut().zip(head_logits) {
                    *p = config.alpha * (h - z) + (1.0 - config.alpha) * *p;
                }
            }
        }
        let name = if config.expected_credit {
            let peak = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = probs.iter().map(|p| (p - peak).exp()).collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>().max(1e-9);
            (0..INSTRUMENTS.len()).max_by(|&a, &b| {
                let sc = |j| (0..INSTRUMENTS.len())
                    .map(|t| credit(j, t) * weights[t] / sum).sum::<f64>();
                sc(a).total_cmp(&sc(b)).then_with(|| b.cmp(&a))
            }).unwrap()
        } else {
            probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0))).unwrap().0
        };
        if config.merge_same_instrument {
            if let Some(&existing) = merged.get(&name) {
                out[existing].note_indices.extend(indices);
                continue;
            }
            merged.insert(name, out.len());
        }
        out.push(InstrumentGroup { instrument_index: name, tracklet_index: tracklet, note_indices: indices });
    }
    for group in &mut out { group.note_indices.sort_unstable(); }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn onehot(k: usize) -> Vec<f32> {
        (0..INSTRUMENT_CLASSES).map(|i| if i == k { 1.0 } else { 0.0 }).collect()
    }
    fn sample() -> (GroupPredictions, Vec<(f64, f64, u8)>) {
        let notes = vec![(0.1, 0.3, 60), (0.3, 0.6, 64), (0.2, 0.5, 40), (0.5, 0.8, 41)];
        let embeddings = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let probabilities = vec![onehot(0), onehot(0), onehot(28), onehot(28)];
        let windows = vec![WindowPrediction {
            start_seconds: 0.0, note_indices: vec![0, 1, 2, 3],
            embeddings: embeddings.clone(), probabilities: probabilities.clone(),
        }];
        (GroupPredictions { window_seconds: 4.0, embeddings, probabilities, hidden: None, windows }, notes)
    }
    #[test]
    fn distinguishes_piano_from_electric_bass() {
        let (p, notes) = sample();
        let cfg = GroupConfig { min_notes: 1, refine: 0, absorb: 0.0, ..Default::default() };
        let groups = decode_groups(&p, &notes, cfg, None).unwrap();
        assert_eq!(groups.len(), 2);
        assert!(groups.iter().any(|g| g.note_indices == [0, 1]));
        assert!(groups.iter().any(|g| g.note_indices == [2, 3]));
    }
    #[test]
    fn no_predictions_does_not_hallucinate_tracks() {
        let p = GroupPredictions { window_seconds: 4.0, embeddings: vec![], probabilities: vec![], hidden: None, windows: vec![] };
        assert!(decode_groups(&p, &[], GroupConfig::default(), None).unwrap().is_empty());
    }
}
