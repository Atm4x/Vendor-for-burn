use std::collections::BTreeMap;

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

pub(crate) fn normalize(a: &[f32]) -> Vec<f64> {
    let norm = (a.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()).sqrt().max(1e-9);
    a.iter().map(|&x| f64::from(x) / norm).collect()
}

pub(crate) fn normalized64(a: &[f64]) -> Vec<f64> {
    let norm = dot(a, a).sqrt().max(1e-9);
    a.iter().map(|x| x / norm).collect()
}

pub(crate) fn cosine(a: &[f64], b: &[f64]) -> f64 { dot(a, b) }

pub fn average_linkage(distances: &[Vec<f64>], threshold: f64) -> Result<Vec<usize>, String> {
    let n = distances.len();
    if distances.iter().any(|row| row.len() != n)
        || distances.iter().flatten().any(|x| !x.is_finite()) || !threshold.is_finite()
    { return Err("invalid hierarchical clustering distance matrix".into()); }
    if n == 0 { return Ok(Vec::new()); }
    let mut d = distances.to_vec();
    let mut size = vec![1usize; n];
    let mut alive = vec![true; n];
    let mut parent = (0..n).collect::<Vec<_>>();
    loop {
        let mut closest = None;
        for i in 0..n {
            if !alive[i] { continue; }
            for j in i + 1..n {
                if !alive[j] { continue; }
                let distance = d[i][j];
                if closest.map(|(_, _, best)| distance < best).unwrap_or(true) {
                    closest = Some((i, j, distance));
                }
            }
        }
        let Some((i, j, score)) = closest else { break; };
        if score > threshold { break; }
        for k in 0..n {
            if !alive[k] || i == k || j == k { continue; }
            let mean = (d[i][k] * size[i] as f64 + d[j][k] * size[j] as f64)
                / (size[i] + size[j]) as f64;
            d[i][k] = mean;
            d[k][i] = mean;
        }
        size[i] += size[j];
        alive[j] = false;
        for id in &mut parent { if *id == j { *id = i; } }
    }
    let mut label = BTreeMap::new();
    Ok(parent.into_iter().map(|id| {
        if let Some(&label) = label.get(&id) { label } else {
            let next = label.len();
            label.insert(id, next);
            next
        }
    }).collect())
}

pub fn maximum_matching(matrix: &[Vec<usize>]) -> Result<Vec<(usize, usize)>, String> {
    let rows = matrix.len();
    if rows == 0 { return Ok(Vec::new()); }
    let cols = matrix[0].len();
    if matrix.iter().any(|row| row.len() != cols) { return Err("ragged overlap matrix".into()); }
    if cols == 0 { return Ok(Vec::new()); }
    let transposed = rows > cols;
    let (n, m) = if transposed { (cols, rows) } else { (rows, cols) };
    let score = |i: usize, j: usize| -> i64 {
        if transposed { matrix[j][i] as i64 } else { matrix[i][j] as i64 }
    };
    let mut u = vec![0i64; n + 1];
    let mut v = vec![0i64; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![i64::MAX / 4; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = i64::MAX / 4;
            let mut j1 = 0;
            for j in 1..=m {
                if used[j] { continue; }
                let cur = -score(i0 - 1, j - 1) - u[i0] - v[j];
                if cur < minv[j] { minv[j] = cur; way[j] = j0; }
                if minv[j] < delta { delta = minv[j]; j1 = j; }
            }
            for j in 0..=m {
                if used[j] { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
            if p[j0] == 0 { break; }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 { break; }
        }
    }
    let mut pairs = Vec::new();
    for j in 1..=m {
        if p[j] != 0 {
            let (a, b) = if transposed { (j - 1, p[j] - 1) } else { (p[j] - 1, j - 1) };
            pairs.push((a, b));
        }
    }
    pairs.sort_unstable();
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn average_linkage_merges_neighbor_pairs() {
        let d = vec![
            vec![0.0, 0.05, 0.9, 0.9],
            vec![0.05, 0.0, 0.9, 0.9],
            vec![0.9, 0.9, 0.0, 0.05],
            vec![0.9, 0.9, 0.05, 0.0],
        ];
        let labels = average_linkage(&d, 0.2).unwrap();
        assert_eq!(labels, [0, 0, 1, 1]);
    }
    #[test]
    fn hungarian_assignment_maximizes_global_overlap() {
        let m = vec![vec![10, 8], vec![9, 0]];
        assert_eq!(maximum_matching(&m).unwrap(), [(0, 1), (1, 0)]);
    }
    #[test]
    fn matching_supports_rectangular_matrices() {
        let m = vec![vec![2, 3], vec![9, 0], vec![1, 8]];
        assert_eq!(maximum_matching(&m).unwrap(), [(1, 0), (2, 1)]);
    }
}
