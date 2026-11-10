use cubecl::{prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubecl_server::runtime::Runtime;
use cubek_convolution::kernels::forward::{
    launch_pack_conv1d_weights, launch_prepacked_temporal_pair,
};
use cubek_convolution::{ConvolutionArgs, DirectTensors, launch_direct};
use serde::Serialize;
use std::{fs, hint::black_box, path::PathBuf, time::Instant};

#[derive(Serialize)]
struct CaseResult {
    name: String,
    cin: usize,
    cout: usize,
    length: usize,
    kernel: usize,
    dilation: usize,
    pack_ms: f64,
    direct_ms: f64,
    packed_ms: f64,
    speedup_direct_over_packed: f64,
    max_abs_error: f32,
    max_normalized_error: f32,
    iterations: usize,
    passed: bool,
    direct_probe: [f32; 3],
    output_sum: f64,
}

#[derive(Serialize)]
struct Report {
    tool: &'static str,
    backend: String,
    status: &'static str,
    semantics: &'static str,
    results: Vec<CaseResult>,
}

fn tensor(client: &Client, data: &[f32], shape: impl Into<Shape>) -> TensorHandle {
    TensorHandle::new_contiguous(
        shape,
        client.create_from_slice(bytemuck::cast_slice(data)),
        f32::elem_type_native(),
    )
}
fn read(client: &Client, t: TensorHandle) -> Vec<f32> {
    let bytes = client.read_one_unchecked(t.handle);
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}
fn run_case(
    client: &Client,
    name: &str,
    cin: usize,
    cout: usize,
    length: usize,
    k: usize,
    d: usize,
    warmup: usize,
    iterations: usize,
) -> Result<CaseResult, String> {
    let pad = (k - 1) * d / 2;
    let input: Vec<f32> = (0..length * cin)
        .map(|i| ((i * 29 % 311) as f32 - 150.) / 211.)
        .collect();
    let weights: Vec<f32> = (0..cout * k * cin)
        .map(|i| ((i * 13 % 199) as f32 - 100.) / 379.)
        .collect();
    let x = tensor(client, &input, Shape::new([1, length, cin]));
    let w = tensor(client, &weights, Shape::new([cout, k, cin]));
    let wp = TensorHandle::empty(
        client,
        Shape::new([1, k, cin, cout]),
        f32::elem_type_native(),
    );
    let clock = Instant::now();
    launch_pack_conv1d_weights(
        client,
        w.clone().binding(),
        wp.clone().binding(),
        f32::elem_type_native(),
    )
    .map_err(|e| format!("pack kernel failed: {e:?}"))?;
    let mut packed_expect = vec![0f32; weights.len()];
    for kk in 0..k {
        for ci in 0..cin {
            for co in 0..cout {
                packed_expect[(kk * cin + ci) * cout + co] = weights[(co * k + kk) * cin + ci];
            }
        }
    }
    let packed_actual = read(client, wp.clone());
    if packed_actual != packed_expect {
        return Err("packed weights do not match reference exactly".into());
    }
    let pack_ms = clock.elapsed().as_secs_f64() * 1000.0;
    let opts = ConvolutionArgs {
        stride: [1],
        padding: [pad],
        dilation: [d],
    };
    let mut direct = Vec::new();
    let mut packed = Vec::new();
    let mut direct_out = Vec::new();
    let mut packed_out = Vec::new();
    for repeat in 0..(warmup + iterations) {
        for variant in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
            let out = TensorHandle::empty(
                client,
                Shape::new([1, length, cout]),
                f32::elem_type_native(),
            );
            let begin = Instant::now();
            let tensors = DirectTensors {
                input: x.clone().binding(),
                weight: if variant == 0 {
                    w.clone().binding()
                } else {
                    wp.clone().binding()
                },
                bias: None,
                out: out.clone().binding(),
            };
            if variant == 0 {
                launch_direct::<1>(client, tensors, opts.clone(), 1, f32::elem_type_native())
                    .map_err(|e| format!("direct failed: {e:?}"))?;
            } else {
                launch_prepacked_temporal_pair(
                    client,
                    tensors,
                    opts.clone(),
                    1,
                    f32::elem_type_native(),
                )
                .map_err(|e| format!("packed failed: {e:?}"))?;
            }
            let values = read(client, out);
            black_box(&values);
            let elapsed = begin.elapsed().as_secs_f64() * 1000.0;
            if repeat >= warmup {
                if variant == 0 {
                    direct.push(elapsed);
                } else {
                    packed.push(elapsed);
                }
            }
            if repeat == warmup {
                if variant == 0 {
                    direct_out = values;
                } else {
                    packed_out = values;
                }
            }
        }
    }
    if direct_out.len() != packed_out.len() {
        return Err("output sizes differ".into());
    }
    let mut max_abs_error = 0.0f32;
    let mut max_normalized_error = 0.0f32;
    for (a, b) in direct_out.iter().zip(&packed_out) {
        let diff = (*a - *b).abs();
        max_abs_error = max_abs_error.max(diff);
        max_normalized_error = max_normalized_error.max(diff / (1.0 + a.abs()));
    }
    let direct_ms = median(&mut direct);
    let packed_ms = median(&mut packed);
    let passed = max_normalized_error <= 0.0002 && direct_ms.is_finite() && packed_ms.is_finite();
    let direct_probe = [
        direct_out[0],
        direct_out[direct_out.len() / 2],
        direct_out[direct_out.len() - 1],
    ];
    let output_sum = direct_out.iter().map(|&x| x as f64).sum();
    Ok(CaseResult {
        name: name.into(),
        cin,
        cout,
        length,
        kernel: k,
        dilation: d,
        pack_ms,
        direct_ms,
        packed_ms,
        speedup_direct_over_packed: direct_ms / packed_ms,
        max_abs_error,
        max_normalized_error,
        iterations,
        passed,
        direct_probe,
        output_sum,
    })
}

fn run(client: &Client, backend: &str, warmup: usize, iterations: usize) -> Report {
    let specs = [
        ("NSF K3", 256, 256, 240, 3, 1),
        ("NSF K11 D5", 128, 128, 2400, 11, 5),
        ("NSF K11", 32, 32, 9600, 11, 1),
    ];
    let mut results = Vec::new();
    for (name, cin, cout, len, k, d) in specs {
        eprintln!("[{backend}] benchmarking {name}");
        match run_case(client, name, cin, cout, len, k, d, warmup, iterations) {
            Ok(result) => {
                eprintln!(
                    "[{backend}] {name}: direct {:.3} ms, packed {:.3} ms, error {:.6}",
                    result.direct_ms, result.packed_ms, result.max_abs_error
                );
                results.push(result);
            }
            Err(err) => {
                eprintln!("[{backend}] {name}: {err}");
                return Report {
                    tool: "rvc-cubecl-gpu-bench",
                    backend: backend.into(),
                    status: "error",
                    semantics: "launch + synchronous CPU readback; warmed median; model shapes but deterministic synthetic data",
                    results,
                };
            }
        }
    }
    let status = if results.iter().all(|x| x.passed) {
        "pass"
    } else {
        "parity_failed"
    };
    Report {
        tool: "rvc-cubecl-gpu-bench",
        backend: backend.into(),
        status,
        semantics: "launch + synchronous CPU readback; warmed median; model shapes but deterministic synthetic data",
        results,
    }
}
fn usage() {
    println!(
        "rvc-cubecl-gpu-bench --backend wgpu|cuda --warmup 3 --iterations 10 --json report.json\nGPU backends are compiled via Cargo features. CUDA Toolkit is not installed by this program.\nMeasured time includes kernel dispatch and synchronous CPU readback (not device-only kernel time)."
    );
}
fn main() {
    let mut args = std::env::args().skip(1);
    let mut backend = "wgpu".to_owned();
    let mut iterations = 10usize;
    let mut warmup = 3usize;
    let mut json = None::<PathBuf>;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "-h" | "--help" => {
                usage();
                return;
            }
            "--backend" => backend = args.next().expect("--backend value"),
            "--iterations" => {
                iterations = args
                    .next()
                    .expect("--iterations number")
                    .parse()
                    .expect("iterations")
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .expect("--warmup number")
                    .parse()
                    .expect("warmup")
            }
            "--json" => json = Some(args.next().expect("--json filename").into()),
            _ => {
                eprintln!("Unknown option: {flag}");
                usage();
                std::process::exit(2);
            }
        }
    }
    if iterations == 0 || iterations > 1000 || warmup > 100 {
        eprintln!("iterations must be 1..1000, warmup 0..100");
        std::process::exit(2);
    }
    let result = match backend.as_str() {
        #[cfg(feature = "wgpu")]
        "wgpu" => {
            use cubecl_wgpu::{WgpuRuntime, WgslCompiler};
            if !WgpuRuntime::<WgslCompiler>::is_available() {
                eprintln!("No hardware WGPU adapter available");
                std::process::exit(3);
            }
            let client = WgpuRuntime::<WgslCompiler>::client(&Default::default());
            run(&client, &backend, warmup, iterations)
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            use cubecl_cuda::CudaRuntime;
            if !CudaRuntime::is_available() {
                eprintln!("No CUDA driver/device available");
                std::process::exit(3);
            }
            let client = CudaRuntime::client(&Default::default());
            run(&client, &backend, warmup, iterations)
        }
        _ => {
            eprintln!("Backend '{backend}' was not compiled in or is unknown");
            usage();
            std::process::exit(2);
        }
    };
    let content = serde_json::to_string_pretty(&result).unwrap();
    println!("{content}");
    if let Some(path) = json {
        fs::write(path, format!("{content}\n")).expect("failed to write report");
    }
    if result.status != "pass" {
        std::process::exit(1);
    }
}
