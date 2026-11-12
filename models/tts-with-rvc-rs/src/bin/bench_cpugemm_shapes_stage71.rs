#[cfg(feature = "cpu")]
fn main() -> Result<(), String> {
    use burn::tensor::{Device, Tensor, TensorData};
    use serde::Serialize;
    use std::time::Instant;

    #[derive(Clone, Copy)]
    struct Case {
        name: &'static str,
        batch: usize,
        rhs_batch: usize,
        m: usize,
        k: usize,
        n: usize,
        lhs_transposed: bool,
        rhs_transposed: bool,
    }
    #[derive(Serialize)]
    struct ResultRow {
        case: &'static str,
        batch: usize,
        rhs_batch: usize,
        m: usize,
        k: usize,
        n: usize,
        lhs_transposed: bool,
        rhs_transposed: bool,
        cold_ms: f64,
        warm_ms: f64,
        max_abs_sampled_error: f64,
        sampled_elements: usize,
        bitwise_repeat: bool,
    }
    fn value(index: usize, seed: usize) -> f32 {
        (((index.wrapping_mul(37).wrapping_add(seed * 71)) % 1031) as f32 - 515.0) / 2048.0
    }
    let cases = [
        Case {
            name: "hubert_qkv_12",
            batch: 1,
            rhs_batch: 1,
            m: 12,
            k: 768,
            n: 768,
            lhs_transposed: false,
            rhs_transposed: true,
        },
        Case {
            name: "hubert_ffn_100",
            batch: 1,
            rhs_batch: 1,
            m: 100,
            k: 768,
            n: 3072,
            lhs_transposed: false,
            rhs_transposed: true,
        },
        Case {
            name: "hubert_ffn2_384",
            batch: 1,
            rhs_batch: 1,
            m: 384,
            k: 3072,
            n: 768,
            lhs_transposed: false,
            rhs_transposed: true,
        },
        Case {
            name: "hubert_qkv_900",
            batch: 1,
            rhs_batch: 1,
            m: 900,
            k: 768,
            n: 768,
            lhs_transposed: false,
            rhs_transposed: true,
        },
        Case {
            name: "rvc_text_encoder",
            batch: 1,
            rhs_batch: 1,
            m: 27,
            k: 256,
            n: 768,
            lhs_transposed: false,
            rhs_transposed: false,
        },
        Case {
            name: "rmvpe_projection",
            batch: 1,
            rhs_batch: 1,
            m: 33,
            k: 768,
            n: 1536,
            lhs_transposed: false,
            rhs_transposed: true,
        },
        Case {
            name: "irregular_both_transposed",
            batch: 1,
            rhs_batch: 1,
            m: 13,
            k: 31,
            n: 27,
            lhs_transposed: true,
            rhs_transposed: true,
        },
        Case {
            name: "irregular_unaligned",
            batch: 1,
            rhs_batch: 1,
            m: 45,
            k: 769,
            n: 513,
            lhs_transposed: true,
            rhs_transposed: false,
        },
        Case {
            name: "batched_attention",
            batch: 12,
            rhs_batch: 12,
            m: 17,
            k: 64,
            n: 19,
            lhs_transposed: false,
            rhs_transposed: true,
        },
        Case {
            name: "broadcast_rhs_attention",
            batch: 12,
            rhs_batch: 1,
            m: 23,
            k: 64,
            n: 21,
            lhs_transposed: true,
            rhs_transposed: false,
        },
        Case {
            name: "large_irregular",
            batch: 1,
            rhs_batch: 1,
            m: 67,
            k: 1025,
            n: 259,
            lhs_transposed: false,
            rhs_transposed: true,
        },
    ];
    let dev = Device::cpu();
    for case in cases {
        let a_data: Vec<f32> = (0..case.batch * case.m * case.k)
            .map(|i| value(i, 11))
            .collect();
        let b_data: Vec<f32> = (0..case.rhs_batch * case.k * case.n)
            .map(|i| value(i, 17))
            .collect();
        let a = if case.lhs_transposed {
            Tensor::<3>::from_data(
                TensorData::new(a_data.clone(), [case.batch, case.k, case.m]),
                &dev,
            )
            .swap_dims(1, 2)
        } else {
            Tensor::<3>::from_data(
                TensorData::new(a_data.clone(), [case.batch, case.m, case.k]),
                &dev,
            )
        };
        let b = if case.rhs_transposed {
            Tensor::<3>::from_data(
                TensorData::new(b_data.clone(), [case.rhs_batch, case.n, case.k]),
                &dev,
            )
            .swap_dims(1, 2)
        } else {
            Tensor::<3>::from_data(
                TensorData::new(b_data.clone(), [case.rhs_batch, case.k, case.n]),
                &dev,
            )
        };
        let compute = || -> Result<Vec<f32>, String> {
            a.clone()
                .matmul(b.clone())
                .into_data()
                .try_into_vec::<f32>()
                .map_err(|e| format!("{}: readback failed: {e:?}", case.name))
        };
        let start = Instant::now();
        let first = compute()?;
        let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let second = compute()?;
        let warm_ms = start.elapsed().as_secs_f64() * 1000.0;
        if first.len() != case.batch * case.m * case.n || second.len() != first.len() {
            return Err(format!("{}: wrong output shape", case.name));
        }
        let bitwise_repeat = first
            .iter()
            .zip(&second)
            .all(|(x, y)| x.to_bits() == y.to_bits());
        let mut max_abs = 0.0f64;
        let mut samples = 0;
        for idx in 0..24 {
            let batch = idx * 7 % case.batch;
            let row = (idx * 83 + 5) % case.m;
            let col = (idx * 97 + 13) % case.n;
            let bi = if case.rhs_batch == 1 { 0 } else { batch };
            let mut reference = 0.0f64;
            for kk in 0..case.k {
                let ai = if case.lhs_transposed {
                    (batch * case.k + kk) * case.m + row
                } else {
                    (batch * case.m + row) * case.k + kk
                };
                let bj = if case.rhs_transposed {
                    (bi * case.n + col) * case.k + kk
                } else {
                    (bi * case.k + kk) * case.n + col
                };
                reference += a_data[ai] as f64 * b_data[bj] as f64;
            }
            let got = second[(batch * case.m + row) * case.n + col] as f64;
            if !got.is_finite() {
                return Err(format!("{}: nonfinite output", case.name));
            }
            max_abs = max_abs.max((got - reference).abs());
            samples += 1;
        }
        let result = ResultRow {
            case: case.name,
            batch: case.batch,
            rhs_batch: case.rhs_batch,
            m: case.m,
            k: case.k,
            n: case.n,
            lhs_transposed: case.lhs_transposed,
            rhs_transposed: case.rhs_transposed,
            cold_ms,
            warm_ms,
            max_abs_sampled_error: max_abs,
            sampled_elements: samples,
            bitwise_repeat,
        };
        println!(
            "{}",
            serde_json::to_string(&result).map_err(|e| e.to_string())?
        );
        if !bitwise_repeat || max_abs > 0.002 {
            return Err(format!(
                "{}: numerical regression; error={max_abs}, deterministic={bitwise_repeat}",
                case.name
            ));
        }
    }
    Ok(())
}
#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("CPU feature required");
    std::process::exit(2);
}
