use std::{hint::black_box, time::Instant};

const G: [[f32; 3]; 4] = [
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.0, 0.0, 1.0],
];
const BT: [[f32; 4]; 4] = [
    [1.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, -1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, -1.0],
];
const AT: [[f32; 4]; 2] = [[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0]];

#[derive(Clone, Copy, Debug)]
struct Shape {
    batch: usize,
    height: usize,
    width: usize,
    cin: usize,
    cout: usize,
    groups: usize,
}

impl Shape {
    fn in_group(self) -> usize {
        self.cin / self.groups
    }
    fn out_group(self) -> usize {
        self.cout / self.groups
    }
    fn input_len(self) -> usize {
        self.batch * self.cin * self.height * self.width
    }
    fn output_len(self) -> usize {
        self.batch * self.cout * self.height * self.width
    }
    fn weight_len(self) -> usize {
        self.cout * self.in_group() * 9
    }
    fn x_index(self, b: usize, c: usize, y: usize, x: usize) -> usize {
        ((b * self.cin + c) * self.height + y) * self.width + x
    }
    fn o_index(self, b: usize, c: usize, y: usize, x: usize) -> usize {
        ((b * self.cout + c) * self.height + y) * self.width + x
    }
    fn w_index(self, oc: usize, ic: usize, ky: usize, kx: usize) -> usize {
        (oc * self.in_group() + ic) * 9 + ky * 3 + kx
    }
}

fn random(seed: &mut u64, scale: f32) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 32) as f32 / 4_294_967_295.0 - 0.5) * 2.0 * scale
}

fn direct(s: Shape, input: &[f32], weights: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; s.output_len()];
    for b in 0..s.batch {
        for oc in 0..s.cout {
            let group = oc / s.out_group();
            for y in 0..s.height {
                for x in 0..s.width {
                    let mut sum = bias[oc];
                    for ic in 0..s.in_group() {
                        for ky in 0..3 {
                            let iy = y as isize + ky as isize - 1;
                            if iy < 0 || iy >= s.height as isize {
                                continue;
                            }
                            for kx in 0..3 {
                                let ix = x as isize + kx as isize - 1;
                                if ix < 0 || ix >= s.width as isize {
                                    continue;
                                }
                                sum += input[s.x_index(
                                    b,
                                    group * s.in_group() + ic,
                                    iy as usize,
                                    ix as usize,
                                )] * weights[s.w_index(oc, ic, ky, kx)];
                            }
                        }
                    }
                    output[s.o_index(b, oc, y, x)] = sum;
                }
            }
        }
    }
    output
}

fn transpose3x3_kernel(weights: &[f32], start: usize) -> [f32; 16] {
    let mut tmp = [[0.0f32; 3]; 4];
    for y in 0..4 {
        for x in 0..3 {
            for k in 0..3 {
                tmp[y][x] += G[y][k] * weights[start + k * 3 + x];
            }
        }
    }
    let mut out = [0.0f32; 16];
    for y in 0..4 {
        for x in 0..4 {
            for k in 0..3 {
                out[y * 4 + x] += tmp[y][k] * G[x][k];
            }
        }
    }
    out
}

fn transform_input(tile: [[f32; 4]; 4]) -> [f32; 16] {
    let mut tmp = [[0.0f32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            for k in 0..4 {
                tmp[y][x] += BT[y][k] * tile[k][x];
            }
        }
    }
    let mut out = [0.0f32; 16];
    for y in 0..4 {
        for x in 0..4 {
            for k in 0..4 {
                out[y * 4 + x] += tmp[y][k] * BT[x][k];
            }
        }
    }
    out
}

fn inverse_transform(m: &[f32; 16], bias: f32) -> [f32; 4] {
    let mut tmp = [[0.0f32; 4]; 2];
    for y in 0..2 {
        for x in 0..4 {
            for k in 0..4 {
                tmp[y][x] += AT[y][k] * m[k * 4 + x];
            }
        }
    }
    let mut out = [bias; 4];
    for y in 0..2 {
        for x in 0..2 {
            for k in 0..4 {
                out[y * 2 + x] += tmp[y][k] * AT[x][k];
            }
        }
    }
    out
}

struct Winograd3x3 {
    transformed_weights: Vec<f32>,
    shape: Shape,
}

impl Winograd3x3 {
    fn prepack(shape: Shape, weights: &[f32]) -> Self {
        assert!(
            shape.groups > 0
                && shape.cin.is_multiple_of(shape.groups)
                && shape.cout.is_multiple_of(shape.groups)
        );
        assert_eq!(weights.len(), shape.weight_len());
        let mut transformed_weights = vec![0.0; shape.cout * shape.in_group() * 16];
        for oc in 0..shape.cout {
            for ic in 0..shape.in_group() {
                let out = transpose3x3_kernel(weights, shape.w_index(oc, ic, 0, 0));
                let offset = (oc * shape.in_group() + ic) * 16;
                transformed_weights[offset..offset + 16].copy_from_slice(&out);
            }
        }
        Self {
            transformed_weights,
            shape,
        }
    }

    fn apply(&self, input: &[f32], bias: &[f32]) -> Vec<f32> {
        let s = self.shape;
        assert_eq!(input.len(), s.input_len());
        assert_eq!(bias.len(), s.cout);
        let mut output = vec![0.0; s.output_len()];
        let mut transformed_input = vec![0.0f32; s.in_group() * 16];
        for b in 0..s.batch {
            for group in 0..s.groups {
                for ty in 0..s.height.div_ceil(2) {
                    for tx in 0..s.width.div_ceil(2) {
                        for ic in 0..s.in_group() {
                            let mut tile = [[0.0; 4]; 4];
                            for y in 0..4 {
                                let iy = ty as isize * 2 + y as isize - 1;
                                if iy < 0 || iy >= s.height as isize {
                                    continue;
                                }
                                for x in 0..4 {
                                    let ix = tx as isize * 2 + x as isize - 1;
                                    if ix >= 0 && ix < s.width as isize {
                                        tile[y][x] = input[s.x_index(
                                            b,
                                            group * s.in_group() + ic,
                                            iy as usize,
                                            ix as usize,
                                        )];
                                    }
                                }
                            }
                            let v = transform_input(tile);
                            transformed_input[ic * 16..(ic + 1) * 16].copy_from_slice(&v);
                        }
                        for oc_local in 0..s.out_group() {
                            let oc = group * s.out_group() + oc_local;
                            let mut m = [0.0f32; 16];
                            for ic in 0..s.in_group() {
                                let u = &self.transformed_weights[(oc * s.in_group() + ic) * 16
                                    ..(oc * s.in_group() + ic + 1) * 16];
                                let v = &transformed_input[ic * 16..(ic + 1) * 16];
                                for j in 0..16 {
                                    m[j] += u[j] * v[j];
                                }
                            }
                            let values = inverse_transform(&m, bias[oc]);
                            for dy in 0..2 {
                                let y = ty * 2 + dy;
                                if y >= s.height {
                                    continue;
                                }
                                for dx in 0..2 {
                                    let x = tx * 2 + dx;
                                    if x < s.width {
                                        output[s.o_index(b, oc, y, x)] = values[dy * 2 + dx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        output
    }
}

fn compare(shape: Shape, scale: f32, check_speed: bool) {
    let mut seed = 0x1234_5678_90ab_cdef ^ (shape.cin as u64 * 17 + shape.cout as u64);
    let input: Vec<_> = (0..shape.input_len())
        .map(|_| random(&mut seed, scale))
        .collect();
    let weight: Vec<_> = (0..shape.weight_len())
        .map(|_| random(&mut seed, scale))
        .collect();
    let bias: Vec<_> = (0..shape.cout).map(|_| random(&mut seed, scale)).collect();
    let packed = Winograd3x3::prepack(shape, &weight);
    let ref_output = direct(shape, &input, &weight, &bias);
    let output = packed.apply(&input, &bias);
    let mut err2 = 0.0f64;
    let mut sig2 = 0.0f64;
    let mut max = 0.0f64;
    for (expected, found) in ref_output.iter().zip(&output) {
        let error = (expected - found) as f64;
        err2 += error * error;
        sig2 += (*expected as f64) * (*expected as f64);
        max = max.max(error.abs());
    }
    let rms = (err2 / output.len() as f64).sqrt();
    let snr = if err2 == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sig2 / err2).log10()
    };
    assert!(
        snr > 75.0 || max < (0.00001 * scale.max(1.0)) as f64,
        "bad precision {shape:?}: max={max} snr={snr}"
    );
    println!("PASS {shape:?}, scale={scale}, max_abs={max:.8}, rms={rms:.8}, snr_db={snr:.2}");
    if check_speed {
        black_box(direct(shape, &input, &weight, &bias));
        black_box(packed.apply(&input, &bias));
        let mut direct_times = Vec::new();
        let mut winograd_times = Vec::new();
        for _ in 0..9 {
            let start = Instant::now();
            black_box(direct(shape, &input, &weight, &bias));
            direct_times.push(start.elapsed().as_secs_f64() * 1000.0);
            let start = Instant::now();
            black_box(packed.apply(&input, &bias));
            winograd_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        direct_times.sort_by(f64::total_cmp);
        winograd_times.sort_by(f64::total_cmp);
        let direct_med = direct_times[4];
        let winograd_med = winograd_times[4];
        println!(
            "  CPU warm median of 9, direct_ms={direct_med:.3} winograd_ms={winograd_med:.3} speedup={:.3}x prepacked=true",
            direct_med / winograd_med
        );
    }
}

fn main() {
    for shape in [
        Shape {
            batch: 1,
            height: 1,
            width: 1,
            cin: 1,
            cout: 1,
            groups: 1,
        },
        Shape {
            batch: 1,
            height: 2,
            width: 3,
            cin: 3,
            cout: 7,
            groups: 1,
        },
        Shape {
            batch: 2,
            height: 7,
            width: 9,
            cin: 8,
            cout: 8,
            groups: 2,
        },
        Shape {
            batch: 1,
            height: 13,
            width: 15,
            cin: 8,
            cout: 8,
            groups: 8,
        },
        Shape {
            batch: 1,
            height: 32,
            width: 32,
            cin: 16,
            cout: 32,
            groups: 1,
        },
        Shape {
            batch: 1,
            height: 64,
            width: 64,
            cin: 32,
            cout: 32,
            groups: 1,
        },
    ] {
        compare(shape, 1.0, shape.height >= 32);
    }
    compare(
        Shape {
            batch: 1,
            height: 5,
            width: 7,
            cin: 3,
            cout: 5,
            groups: 1,
        },
        100.0,
        false,
    );
}
