fn conv(
    x: &[f32],
    w: &[f32],
    batch: usize,
    cin: usize,
    cout: usize,
    len: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    pad: usize,
    groups: usize,
    tile: usize,
    fast_interior: bool,
) -> Vec<f32> {
    let effective = dilation * (kernel - 1) + 1;
    let out_len = (len + 2 * pad - effective) / stride + 1;
    let cin_g = cin / groups;
    let cout_g = cout / groups;
    let mut out = vec![0.0; batch * out_len * cout];
    for b in 0..batch {
        for g in 0..groups {
            for oc in 0..cout_g {
                for start in (0..out_len).step_by(tile) {
                    let mut sums = vec![0f32; tile];
                    let first = start as isize * stride as isize - pad as isize;
                    let last = (start + tile - 1) as isize * stride as isize
                        + (kernel - 1) as isize * dilation as isize
                        - pad as isize;
                    let fully_interior = fast_interior
                        && start + tile <= out_len
                        && first >= 0
                        && last < len as isize;
                    for k in 0..kernel {
                        for ci in 0..cin_g {
                            let ww = w[(((g * kernel + k) * cin_g + ci) * cout_g) + oc];
                            for lane in 0..tile {
                                let t = start + lane;
                                if t >= out_len {
                                    continue;
                                }
                                let pos = t as isize * stride as isize + (k * dilation) as isize
                                    - pad as isize;
                                if fully_interior || (pos >= 0 && (pos as usize) < len) {
                                    sums[lane] +=
                                        x[(b * len + pos as usize) * cin + g * cin_g + ci] * ww;
                                }
                            }
                        }
                    }
                    for lane in 0..tile {
                        let t = start + lane;
                        if t < out_len {
                            out[(b * out_len + t) * cout + g * cout_g + oc] = sums[lane];
                        }
                    }
                }
            }
        }
    }
    out
}
fn main() {
    let mut cases = 0;
    for batch in [1usize, 2] {
        for groups in [1usize, 2, 4] {
            for cin_g in [1usize, 2, 3] {
                for cout_g in [1usize, 2, 4] {
                    for len in [7usize, 15, 23] {
                        for kernel in [1usize, 3, 5, 11] {
                            for stride in [1usize, 2] {
                                for dilation in [1usize, 2, 5] {
                                    let cin = cin_g * groups;
                                    let cout = cout_g * groups;
                                    let pad = dilation * (kernel - 1) / 2;
                                    let x: Vec<f32> = (0..batch * len * cin)
                                        .map(|i| ((i * 7 % 41) as f32 - 20.) / 37.)
                                        .collect();
                                    let w: Vec<f32> = (0..groups * kernel * cin_g * cout_g)
                                        .map(|i| ((i * 11 % 67) as f32 - 33.) / 41.)
                                        .collect();
                                    let a = conv(
                                        &x, &w, batch, cin, cout, len, kernel, stride, dilation,
                                        pad, groups, 4, false,
                                    );
                                    let b = conv(
                                        &x, &w, batch, cin, cout, len, kernel, stride, dilation,
                                        pad, groups, 4, true,
                                    );
                                    assert_eq!(a.len(), b.len());
                                    assert_eq!(
                                        a, b,
                                        "batch {batch} groups {groups} cin_g {cin_g} cout_g {cout_g} len {len} k {kernel} stride {stride} dil {dilation}"
                                    );
                                    cases += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "TemporalQuad/TemporalQuadInterior arithmetic reference parity PASS: {cases} configurations"
    );
}
