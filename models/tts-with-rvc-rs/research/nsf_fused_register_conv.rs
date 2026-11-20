//! Experimental output-stationary Conv1d for finite FP32 NSF weights.
//! Uses AVX2/FMA registers directly, with no GEMM, im2col, or intermediate outputs.
//! This is research code and is not dispatched by the production model.
use std::arch::x86_64::*;
use std::{hint::black_box, time::Instant};

fn raw(path: &str) -> Vec<f32> {
    let data = std::fs::read(path).unwrap();
    assert_eq!(data.len() % 4, 0);
    data.chunks_exact(4)
        .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
        .collect()
}

struct SpatialPlan {
    ci: usize,
    co: usize,
    k: usize,
    dilation: usize,
    weights: Vec<f32>,
}
impl SpatialPlan {
    fn new(weights: &[f32], ci: usize, co: usize, k: usize, dilation: usize) -> Self {
        assert_eq!(weights.len(), ci * co * k);
        assert!(k > 0 && k % 2 == 1 && dilation >= 1);
        let blocks = co.div_ceil(4);
        let mut packed = vec![0f32; blocks * ci * k * 4];
        for ob in 0..blocks {
            for c in 0..ci {
                for tap in 0..k {
                    for lane in 0..4 {
                        let oc = ob * 4 + lane;
                        if oc < co {
                            packed[((ob * ci + c) * k + tap) * 4 + lane] =
                                weights[(oc * ci + c) * k + tap];
                        }
                    }
                }
            }
        }
        Self {
            ci,
            co,
            k,
            dilation,
            weights: packed,
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx2(&self, x: &[f32], len: usize) -> Vec<f32> {
        assert_eq!(x.len(), self.ci * len);
        let pad = (self.k - 1) * self.dilation / 2;
        let stride = len + 2 * pad + 8;
        let mut padded = vec![0f32; stride * self.ci];
        for c in 0..self.ci {
            padded[c * stride + pad..c * stride + pad + len]
                .copy_from_slice(&x[c * len..(c + 1) * len]);
        }
        let mut output = vec![0f32; self.co * len];
        for ob in 0..self.co.div_ceil(4) {
            let lanes = (self.co - ob * 4).min(4);
            for start in (0..len).step_by(8) {
                let mut accum = [_mm256_setzero_ps(); 4];
                for c in 0..self.ci {
                    let kernel = (ob * self.ci + c) * self.k * 4;
                    for tap in 0..self.k {
                        let xptr = padded
                            .as_ptr()
                            .add(c * stride + start + tap * self.dilation);
                        let vx = _mm256_loadu_ps(xptr);
                        for lane in 0..lanes {
                            let w = _mm256_set1_ps(self.weights[kernel + tap * 4 + lane]);
                            accum[lane] = _mm256_fmadd_ps(vx, w, accum[lane]);
                        }
                    }
                }
                for lane in 0..lanes {
                    let mut values = [0f32; 8];
                    _mm256_storeu_ps(values.as_mut_ptr(), accum[lane]);
                    let n = (len - start).min(8);
                    output[(ob * 4 + lane) * len + start..][..n].copy_from_slice(&values[..n]);
                }
            }
        }
        output
    }
}

struct ChannelPlan {
    ci: usize,
    co: usize,
    k: usize,
    dilation: usize,
    weights: Vec<f32>,
}
impl ChannelPlan {
    fn new(weights: &[f32], ci: usize, co: usize, k: usize, dilation: usize) -> Self {
        assert_eq!(weights.len(), ci * co * k);
        let blocks = co.div_ceil(8);
        let mut packed = vec![0f32; blocks * ci * k * 8];
        for ob in 0..blocks {
            for c in 0..ci {
                for tap in 0..k {
                    for lane in 0..8 {
                        if ob * 8 + lane < co {
                            packed[((ob * ci + c) * k + tap) * 8 + lane] =
                                weights[((ob * 8 + lane) * ci + c) * k + tap];
                        }
                    }
                }
            }
        }
        Self {
            ci,
            co,
            k,
            dilation,
            weights: packed,
        }
    }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx2(&self, x: &[f32], len: usize) -> Vec<f32> {
        let pad = (self.k - 1) * self.dilation / 2;
        let stride = len + 2 * pad + 8;
        let mut padded = vec![0f32; stride * self.ci];
        for c in 0..self.ci {
            padded[c * stride + pad..c * stride + pad + len]
                .copy_from_slice(&x[c * len..(c + 1) * len]);
        }
        let mut out = vec![0f32; self.co * len];
        for ob in 0..self.co.div_ceil(8) {
            let lanes = (self.co - ob * 8).min(8);
            for t in (0..len).step_by(8) {
                let mut acc = [_mm256_setzero_ps(); 8];
                for c in 0..self.ci {
                    let w_off = (ob * self.ci + c) * self.k * 8;
                    for tap in 0..self.k {
                        let w = _mm256_loadu_ps(self.weights.as_ptr().add(w_off + tap * 8));
                        let input = &padded[c * stride + t + tap * self.dilation..];
                        for i in 0..8 {
                            let a = _mm256_set1_ps(input[i]);
                            acc[i] = _mm256_fmadd_ps(a, w, acc[i]);
                        }
                    }
                }
                for i in 0..(len - t).min(8) {
                    let mut vals = [0f32; 8];
                    _mm256_storeu_ps(vals.as_mut_ptr(), acc[i]);
                    for lane in 0..lanes {
                        out[(ob * 8 + lane) * len + t + i] = vals[lane];
                    }
                }
            }
        }
        out
    }
}

struct ChannelPlan16 {
    ci: usize,
    co: usize,
    k: usize,
    dilation: usize,
    weights: Vec<f32>,
}
impl ChannelPlan16 {
    fn new(weights: &[f32], ci: usize, co: usize, k: usize, dilation: usize) -> Self {
        assert_eq!(weights.len(), ci * co * k);
        assert!(k > 0 && k % 2 == 1 && dilation >= 1);
        let mut packed = vec![0f32; co.div_ceil(16) * ci * k * 16];
        for ob in 0..co.div_ceil(16) {
            for c in 0..ci {
                for tap in 0..k {
                    for lane in 0..16 {
                        if ob * 16 + lane < co {
                            packed[((ob * ci + c) * k + tap) * 16 + lane] =
                                weights[((ob * 16 + lane) * ci + c) * k + tap];
                        }
                    }
                }
            }
        }
        Self {
            ci,
            co,
            k,
            dilation,
            weights: packed,
        }
    }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx2(&self, x: &[f32], len: usize) -> Vec<f32> {
        let pad = (self.k - 1) * self.dilation / 2;
        let stride = len + 2 * pad + 4;
        let mut padded = vec![0f32; stride * self.ci];
        for c in 0..self.ci {
            padded[c * stride + pad..c * stride + pad + len]
                .copy_from_slice(&x[c * len..(c + 1) * len]);
        }
        let mut out = vec![0f32; self.co * len];
        for ob in 0..self.co.div_ceil(16) {
            let lanes = (self.co - ob * 16).min(16);
            for t in (0..len).step_by(4) {
                let mut acc0 = [_mm256_setzero_ps(); 4];
                let mut acc1 = [_mm256_setzero_ps(); 4];
                for c in 0..self.ci {
                    let off = (ob * self.ci + c) * self.k * 16;
                    for tap in 0..self.k {
                        let w0 = _mm256_loadu_ps(self.weights.as_ptr().add(off + tap * 16));
                        let w1 = _mm256_loadu_ps(self.weights.as_ptr().add(off + tap * 16 + 8));
                        let input = &padded[c * stride + t + tap * self.dilation..];
                        for i in 0..4 {
                            let vx = _mm256_set1_ps(input[i]);
                            acc0[i] = _mm256_fmadd_ps(vx, w0, acc0[i]);
                            acc1[i] = _mm256_fmadd_ps(vx, w1, acc1[i]);
                        }
                    }
                }
                for i in 0..(len - t).min(4) {
                    let mut v0 = [0f32; 8];
                    let mut v1 = [0f32; 8];
                    _mm256_storeu_ps(v0.as_mut_ptr(), acc0[i]);
                    _mm256_storeu_ps(v1.as_mut_ptr(), acc1[i]);
                    for lane in 0..lanes {
                        out[(ob * 16 + lane) * len + t + i] =
                            if lane < 8 { v0[lane] } else { v1[lane - 8] };
                    }
                }
            }
        }
        out
    }
}

struct ChannelPlan512 {
    ci: usize,
    co: usize,
    k: usize,
    dilation: usize,
    weights: Vec<f32>,
}
impl ChannelPlan512 {
    fn new(weights: &[f32], ci: usize, co: usize, k: usize, dilation: usize) -> Self {
        assert_eq!(weights.len(), ci * co * k);
        assert!(k > 0 && k % 2 == 1 && dilation >= 1);
        let mut packed = vec![0f32; co.div_ceil(16) * ci * k * 16];
        for ob in 0..co.div_ceil(16) {
            for c in 0..ci {
                for tap in 0..k {
                    for lane in 0..16 {
                        if ob * 16 + lane < co {
                            packed[((ob * ci + c) * k + tap) * 16 + lane] =
                                weights[((ob * 16 + lane) * ci + c) * k + tap];
                        }
                    }
                }
            }
        }
        Self {
            ci,
            co,
            k,
            dilation,
            weights: packed,
        }
    }
    #[target_feature(enable = "avx512f,fma")]
    unsafe fn forward_avx512(&self, x: &[f32], len: usize) -> Vec<f32> {
        let pad = (self.k - 1) * self.dilation / 2;
        let stride = len + 2 * pad + 8;
        let mut padded = vec![0f32; stride * self.ci];
        for c in 0..self.ci {
            padded[c * stride + pad..c * stride + pad + len]
                .copy_from_slice(&x[c * len..(c + 1) * len]);
        }
        let mut out = vec![0f32; self.co * len];
        for ob in 0..self.co.div_ceil(16) {
            let lanes = (self.co - ob * 16).min(16);
            for t in (0..len).step_by(8) {
                let mut acc = [_mm512_setzero_ps(); 8];
                for c in 0..self.ci {
                    let off = (ob * self.ci + c) * self.k * 16;
                    for tap in 0..self.k {
                        let w = _mm512_loadu_ps(self.weights.as_ptr().add(off + tap * 16));
                        let input = &padded[c * stride + t + tap * self.dilation..];
                        for i in 0..8 {
                            let vx = _mm512_set1_ps(input[i]);
                            acc[i] = _mm512_fmadd_ps(vx, w, acc[i]);
                        }
                    }
                }
                for i in 0..(len - t).min(8) {
                    let mut v = [0f32; 16];
                    _mm512_storeu_ps(v.as_mut_ptr(), acc[i]);
                    for lane in 0..lanes {
                        out[(ob * 16 + lane) * len + t + i] = v[lane];
                    }
                }
            }
        }
        out
    }
}

struct ChannelPlan512x32 {
    ci: usize,
    co: usize,
    k: usize,
    dilation: usize,
    weights: Vec<f32>,
}
impl ChannelPlan512x32 {
    fn new(weights: &[f32], ci: usize, co: usize, k: usize, dilation: usize) -> Self {
        assert_eq!(weights.len(), ci * co * k);
        assert!(k > 0 && k % 2 == 1 && dilation >= 1);
        let mut packed = vec![0f32; co.div_ceil(32) * ci * k * 32];
        for ob in 0..co.div_ceil(32) {
            for c in 0..ci {
                for tap in 0..k {
                    for lane in 0..32 {
                        if ob * 32 + lane < co {
                            packed[((ob * ci + c) * k + tap) * 32 + lane] =
                                weights[((ob * 32 + lane) * ci + c) * k + tap];
                        }
                    }
                }
            }
        }
        Self {
            ci,
            co,
            k,
            dilation,
            weights: packed,
        }
    }
    #[target_feature(enable = "avx512f,fma")]
    unsafe fn forward_avx512(&self, x: &[f32], len: usize) -> Vec<f32> {
        let pad = (self.k - 1) * self.dilation / 2;
        let stride = len + 2 * pad + 8;
        let mut padded = vec![0f32; stride * self.ci];
        for c in 0..self.ci {
            padded[c * stride + pad..c * stride + pad + len]
                .copy_from_slice(&x[c * len..(c + 1) * len]);
        }
        let mut out = vec![0f32; self.co * len];
        for ob in 0..self.co.div_ceil(32) {
            let lanes = (self.co - ob * 32).min(32);
            for t in (0..len).step_by(8) {
                let mut acc0 = [_mm512_setzero_ps(); 8];
                let mut acc1 = [_mm512_setzero_ps(); 8];
                for c in 0..self.ci {
                    let off = (ob * self.ci + c) * self.k * 32;
                    for tap in 0..self.k {
                        let w0 = _mm512_loadu_ps(self.weights.as_ptr().add(off + tap * 32));
                        let w1 = _mm512_loadu_ps(self.weights.as_ptr().add(off + tap * 32 + 16));
                        let input = &padded[c * stride + t + tap * self.dilation..];
                        for i in 0..8 {
                            let vx = _mm512_set1_ps(input[i]);
                            acc0[i] = _mm512_fmadd_ps(vx, w0, acc0[i]);
                            acc1[i] = _mm512_fmadd_ps(vx, w1, acc1[i]);
                        }
                    }
                }
                for i in 0..(len - t).min(8) {
                    let mut v0 = [0f32; 16];
                    let mut v1 = [0f32; 16];
                    _mm512_storeu_ps(v0.as_mut_ptr(), acc0[i]);
                    _mm512_storeu_ps(v1.as_mut_ptr(), acc1[i]);
                    for lane in 0..lanes {
                        out[(ob * 32 + lane) * len + t + i] =
                            if lane < 16 { v0[lane] } else { v1[lane - 16] };
                    }
                }
            }
        }
        out
    }
}

fn direct_gemm(
    x: &[f32],
    w: &[f32],
    ci: usize,
    co: usize,
    len: usize,
    k: usize,
    dilation: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; co * len];
    let pad = (k - 1) * dilation / 2;
    for tap in 0..k {
        let offset = tap * dilation;
        let start = pad.saturating_sub(offset).min(len);
        let end = (len + pad).saturating_sub(offset).min(len);
        if start >= end {
            continue;
        }
        let in_start = start + offset - pad;
        unsafe {
            gemm::gemm(
                co,
                end - start,
                ci,
                y.as_mut_ptr().add(start),
                1,
                len as isize,
                true,
                w.as_ptr().add(tap),
                k as isize,
                (ci * k) as isize,
                x.as_ptr().add(in_start),
                1,
                len as isize,
                1f32,
                1f32,
                false,
                false,
                false,
                gemm::Parallelism::None,
            );
        }
    }
    y
}

fn median(mut x: Vec<f64>) -> f64 {
    x.sort_by(f64::total_cmp);
    x[x.len() / 2]
}
fn score(x: &[f32], y: &[f32]) -> (f64, f64) {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).fold((0f64, 0f64), |(max, mean), (&a, &b)| {
        (
            max.max((a - b).abs() as f64),
            mean + (a - b).abs() as f64 / x.len() as f64,
        )
    })
}
fn profile(name: &str, ci: usize, co: usize, k: usize, d: usize, len: usize, dir: &str) {
    let x = raw(&format!("{dir}/x.bin"));
    let w = raw(&format!("{dir}/w.bin"));
    let bias = raw(&format!("{dir}/bias.bin"));
    let torch = raw(&format!("{dir}/out.bin"));
    assert_eq!(x.len(), ci * len);
    assert_eq!(w.len(), co * ci * k);
    let actual = direct_gemm(&x, &w, ci, co, len, k, d);
    let plan = SpatialPlan::new(&w, ci, co, k, d);
    let experimental = unsafe { plan.forward_avx2(&x, len) };
    let (max_err, mean_err) = score(&actual, &experimental);
    let (torch_err, _) = score(
        &torch,
        &experimental
            .iter()
            .enumerate()
            .map(|(i, v)| v + bias[i / len])
            .collect::<Vec<_>>(),
    );
    println!("case={name}, max_gemm_err={max_err:.7}, mean_gemm_err={mean_err:.7}, max_torch_err={torch_err:.7}");
    assert!(max_err < 1e-3);
    let mut baseline = Vec::new();
    let mut candidate = Vec::new();
    for _ in 0..3 {
        black_box(direct_gemm(&x, &w, ci, co, len, k, d));
        black_box(unsafe { plan.forward_avx2(&x, len) });
    }
    let spatial = plan;
    let channel = ChannelPlan::new(&w, ci, co, k, d);
    let channel_result = unsafe { channel.forward_avx2(&x, len) };
    let (channel_err, _) = score(&actual, &channel_result);
    assert!(channel_err < 1e-3);
    let channel16 = ChannelPlan16::new(&w, ci, co, k, d);
    let (channel16_err, _) = score(&actual, &unsafe { channel16.forward_avx2(&x, len) });
    assert!(channel16_err < 1e-3);
    let channel512 = ChannelPlan512::new(&w, ci, co, k, d);
    let (channel512_err, _) = score(&actual, &unsafe { channel512.forward_avx512(&x, len) });
    assert!(channel512_err < 1e-3);
    let channel512x32 = ChannelPlan512x32::new(&w, ci, co, k, d);
    let (channel512x32_err, _) = score(&actual, &unsafe { channel512x32.forward_avx512(&x, len) });
    assert!(channel512x32_err < 1e-3);
    let mut channel_times = Vec::new();
    let mut channel16_times = Vec::new();
    let mut channel512_times = Vec::new();
    let mut channel512x32_times = Vec::new();
    for _ in 0..9 {
        let start = Instant::now();
        black_box(direct_gemm(&x, &w, ci, co, len, k, d));
        baseline.push(start.elapsed().as_secs_f64() * 1000.);
        let start = Instant::now();
        black_box(unsafe { spatial.forward_avx2(&x, len) });
        candidate.push(start.elapsed().as_secs_f64() * 1000.);
        let start = Instant::now();
        black_box(unsafe { channel.forward_avx2(&x, len) });
        channel_times.push(start.elapsed().as_secs_f64() * 1000.);
        let start = Instant::now();
        black_box(unsafe { channel16.forward_avx2(&x, len) });
        channel16_times.push(start.elapsed().as_secs_f64() * 1000.);
        let start = Instant::now();
        black_box(unsafe { channel512.forward_avx512(&x, len) });
        channel512_times.push(start.elapsed().as_secs_f64() * 1000.);
        let start = Instant::now();
        black_box(unsafe { channel512x32.forward_avx512(&x, len) });
        channel512x32_times.push(start.elapsed().as_secs_f64() * 1000.);
    }
    let old = median(baseline);
    let new = median(candidate);
    let ch = median(channel_times);
    let ch16 = median(channel16_times);
    let ch512 = median(channel512_times);
    let ch32 = median(channel512x32_times);
    println!("case={name}, gemm_ms={old:.4}, temporal_ms={new:.4}, channel8_ms={ch:.4}, channel16_ms={ch16:.4}, avx512x16_ms={ch512:.4}, avx512x32_ms={ch32:.4}, speedup_avx512x32={:.3}x, max_avx512x32_err={channel512x32_err:.8}",old/ch32);
}
fn main() {
    let cases = [
        ("K3 early", 256, 256, 3, 1, 240, "resblocks_0_convs1_0"),
        ("K11 middle", 128, 128, 11, 5, 2400, "resblocks_5_convs1_2"),
        ("K11 late", 32, 32, 11, 1, 9600, "resblocks_11_convs2_2"),
    ];
    assert!(
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("fma"),
        "AVX-512F and FMA are required; no implicit fallback"
    );
    let root = std::env::args()
        .nth(1)
        .expect("usage: nsf_fused_register_conv /path/to/real_case");
    for (name, ci, co, k, d, len, dir) in cases {
        profile(name, ci, co, k, d, len, &format!("{root}/{dir}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_direct_avx512_matches_reference_across_edge_cases() {
        assert!(std::arch::is_x86_feature_detected!("avx512f"));
        assert!(std::arch::is_x86_feature_detected!("fma"));
        let mut cases = 0;
        for ci in [1usize, 3, 8] {
            for co in [1usize, 7, 16, 17, 31, 32, 33] {
                for len in [0usize, 1, 2, 7, 8, 9, 16, 17, 31] {
                    for k in [1usize, 3, 5, 11] {
                        for dilation in [1usize, 2, 5] {
                            let x = (0..ci * len)
                                .map(|i| (i as f32 * 0.217).sin() * 0.37)
                                .collect::<Vec<_>>();
                            let weights = (0..ci * co * k)
                                .map(|i| (i as f32 * 0.113).cos() * 0.13)
                                .collect::<Vec<_>>();
                            let expected = direct_gemm(&x, &weights, ci, co, len, k, dilation);
                            let plan16 = ChannelPlan512::new(&weights, ci, co, k, dilation);
                            let plan32 = ChannelPlan512x32::new(&weights, ci, co, k, dilation);
                            let actual16 = unsafe { plan16.forward_avx512(&x, len) };
                            let actual32 = unsafe { plan32.forward_avx512(&x, len) };
                            assert_eq!(actual16.len(), expected.len());
                            assert_eq!(actual32.len(), expected.len());
                            for (n, (&exp, (&a, &b))) in expected
                                .iter()
                                .zip(actual16.iter().zip(&actual32))
                                .enumerate()
                            {
                                let tolerance = (1e-5f32).max(exp.abs() * 1e-5);
                                assert!((exp - a).abs() <= tolerance, "16-lane ci={ci} co={co} len={len} k={k} d={dilation} at {n}: {exp} != {a}");
                                assert!((exp - b).abs() <= tolerance, "32-lane ci={ci} co={co} len={len} k={k} d={dilation} at {n}: {exp} != {b}");
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 2268);
    }
}
