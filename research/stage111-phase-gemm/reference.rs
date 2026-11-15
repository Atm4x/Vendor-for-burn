#[derive(Clone, Copy)]
struct Case {
    batch: usize, cin: usize, cout: usize, time: usize,
    kernel: usize, stride: usize, padding: usize, groups: usize,
}

fn source(index: usize) -> f32 { ((index as f32) * 0.031).sin() * 0.2 }
fn filter(index: usize) -> f32 { ((index as f32) * 0.027).cos() * 0.1 }

fn reference(c: Case) -> Vec<f32> {
    let cg = c.cin / c.groups;
    let og = c.cout / c.groups;
    let out_t = (c.time - 1) * c.stride + c.kernel - 2 * c.padding;
    let mut output = vec![0f32; c.batch * c.cout * out_t];
    for n in 0..c.batch {
        for ci in 0..c.cin {
            let g = ci / cg;
            for t in 0..c.time {
                for oc in 0..og {
                    for k in 0..c.kernel {
                        let y = t * c.stride + k;
                        if y < c.padding || y - c.padding >= out_t { continue; }
                        let target = (n * c.cout + g * og + oc) * out_t + y - c.padding;
                        let w = filter((ci * og + oc) * c.kernel + k);
                        output[target] += source((n * c.cin + ci) * c.time + t) * w;
                    }
                }
            }
        }
    }
    for n in 0..c.batch {
        for oc in 0..c.cout {
            for t in 0..out_t {
                output[(n * c.cout + oc) * out_t + t] += oc as f32 * 0.001;
            }
        }
    }
    output
}

fn phase_matmul(c: Case) -> Vec<f32> {
    let cg = c.cin / c.groups;
    let og = c.cout / c.groups;
    let out_t = (c.time - 1) * c.stride + c.kernel - 2 * c.padding;
    let tceil = out_t.div_ceil(c.stride);
    let taps = c.kernel.div_ceil(c.stride);
    let reduce = cg * taps;
    let mut output = vec![0f32; c.batch * c.cout * out_t];
    for n in 0..c.batch {
        for group in 0..c.groups {
            for phase in 0..c.stride {
                let q = (phase + c.padding) % c.stride;
                let start = (phase + c.padding) / c.stride;
                let mut w = vec![0f32; reduce * og];
                for ci in 0..cg {
                    for tap in 0..taps {
                        let k = q + tap * c.stride;
                        if k < c.kernel {
                            for oc in 0..og {
                                w[(ci * taps + tap) * og + oc] =
                                    filter(((group * cg + ci) * og + oc) * c.kernel + k);
                            }
                        }
                    }
                }
                for u in 0..tceil {
                    let y = phase + u * c.stride;
                    if y >= out_t { continue; }
                    let mut x = vec![0f32; reduce];
                    for ci in 0..cg {
                        for tap in 0..taps {
                            let source_t = start as isize + u as isize - tap as isize;
                            if (0..c.time as isize).contains(&source_t) {
                                x[ci * taps + tap] = source((n * c.cin + group * cg + ci) * c.time + source_t as usize);
                            }
                        }
                    }
                    for oc in 0..og {
                        let mut v = (group * og + oc) as f32 * 0.001;
                        for r in 0..reduce {
                            v += x[r] * w[r * og + oc];
                        }
                        output[(n * c.cout + group * og + oc) * out_t + y] = v;
                    }
                }
            }
        }
    }
    output
}

#[test]
fn phase_batched_matmul_matches_reference() {
    for c in [
        Case { batch:1, cin:3, cout:5, time:5, kernel:6, stride:2, padding:2, groups:1 },
        Case { batch:1, cin:2, cout:3, time:9, kernel:3, stride:1, padding:1, groups:1 },
        Case { batch:2, cin:8, cout:6, time:21, kernel:16, stride:8, padding:4, groups:1 },
        Case { batch:1, cin:8, cout:12, time:19, kernel:20, stride:10, padding:5, groups:2 },
        Case { batch:2, cin:12, cout:8, time:7, kernel:9, stride:4, padding:3, groups:4 },
        Case { batch:1, cin:4, cout:4, time:11, kernel:5, stride:3, padding:1, groups:1 },
        Case { batch:1, cin:6, cout:12, time:13, kernel:7, stride:2, padding:2, groups:3 },
        Case { batch:2, cin:2, cout:2, time:2, kernel:7, stride:8, padding:1, groups:1 },
    ] {
        let a = reference(c);
        let b = phase_matmul(c);
        assert_eq!(a.len(), b.len());
        let error = a.iter().zip(b.iter()).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max);
        assert!(error < 2e-6, "phase GEMM wrong: {error}, stride={}, kernel={}, groups={}", c.stride,c.kernel,c.groups);
    }
}
