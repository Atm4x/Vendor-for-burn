use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::std::tensor::{TensorHandle, into_contiguous};
use cubecl_cpu::CpuRuntime;
use cubecl_server::runtime::Runtime;
use cubek_convolution::{ConvolutionArgs, DirectTensors, launch_direct};
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
struct Record {
    name: &'static str,
    shape: [usize; 4],
    dilation: usize,
    iterations: usize,
    reference_max_abs: f32,
    strided_max_abs: f32,
    resident_contiguous_ms: f64,
    copy_and_direct_ms: f64,
    strided_direct_ms: f64,
    predicted_speedup: f64,
}

fn tensor(client: &Client, data: &[f32], shape: [usize; 3], strides: [usize; 3]) -> TensorHandle {
    TensorHandle::new(
        client.create_from_slice(bytemuck::cast_slice(data)),
        shape,
        strides,
        f32::elem_type_native(),
    )
}

fn read(client: &Client, tensor: &TensorHandle) -> Vec<f32> {
    let bytes = client.read_one_unchecked(tensor.clone().handle);
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn measure(client: &Client, warmup: usize, iterations: usize, mut run: impl FnMut()) -> f64 {
    let mut samples = Vec::with_capacity(iterations);
    for round in 0..warmup + iterations {
        let started = Instant::now();
        run();
        futures_lite::future::block_on(client.sync()).expect("CPU sync failed");
        if round >= warmup {
            samples.push(started.elapsed().as_secs_f64() * 1000.);
        }
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn case(
    client: &Client,
    name: &'static str,
    cin: usize,
    cout: usize,
    len: usize,
    k: usize,
    dilation: usize,
    warmup: usize,
    iterations: usize,
) -> Record {
    let x_data: Vec<f32> = (0..cin * len)
        .map(|i| ((i * 29 % 379) as f32 - 171.) / 109.)
        .collect();
    let w_data: Vec<f32> = (0..cout * cin * k)
        .map(|i| ((i * 17 % 331) as f32 - 149.) / 237.)
        .collect();
    let physical_x = tensor(client, &x_data, [1, len, cin], [cin * len, 1, len]);
    let physical_w = tensor(client, &w_data, [cout, k, cin], [cin * k, 1, k]);
    let x_contiguous = into_contiguous(
        client,
        physical_x.clone().binding(),
        f32::elem_type_native(),
    );
    let w_contiguous = into_contiguous(
        client,
        physical_w.clone().binding(),
        f32::elem_type_native(),
    );
    let out_len = len;
    let out_strided = TensorHandle::empty(client, [1, out_len, cout], f32::elem_type_native());
    let out_contiguous = TensorHandle::empty(client, [1, out_len, cout], f32::elem_type_native());
    let opts = ConvolutionArgs {
        stride: [1],
        padding: [((k - 1) * dilation) / 2],
        dilation: [dilation],
    };
    let execute = |input: &TensorHandle, weight: &TensorHandle, output: &TensorHandle| {
        launch_direct::<1>(
            client,
            DirectTensors {
                input: input.clone().binding(),
                weight: weight.clone().binding(),
                bias: None,
                out: output.clone().binding(),
            },
            opts.clone(),
            1,
            f32::elem_type_native(),
        )
        .expect("direct convolution failed")
    };
    execute(&x_contiguous, &w_contiguous, &out_contiguous);
    execute(&physical_x, &physical_w, &out_strided);
    let ref_data = read(client, &out_contiguous);
    let test_data = read(client, &out_strided);
    let strided_max_abs = test_data
        .iter()
        .zip(&ref_data)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let reference_max_abs = (0..cout)
        .flat_map(|oc| (0..out_len).map(move |t| (oc, t)))
        .map(|(oc, t)| {
            let mut acc = 0f32;
            for kernel in 0..k {
                let pos = t as isize + (kernel * dilation) as isize - opts.padding[0] as isize;
                if pos >= 0 && pos < len as isize {
                    for ci in 0..cin {
                        acc += x_data[ci * len + pos as usize]
                            * w_data[oc * cin * k + ci * k + kernel];
                    }
                }
            }
            (acc - ref_data[t * cout + oc]).abs()
        })
        .fold(0f32, f32::max);
    assert!(
        strided_max_abs <= 4e-4,
        "strided mismatch on {name}: {strided_max_abs}"
    );
    assert!(
        reference_max_abs <= 8e-4,
        "reference mismatch {name}: {reference_max_abs}"
    );
    let resident_contiguous_ms = measure(client, warmup, iterations, || {
        execute(&x_contiguous, &w_contiguous, &out_contiguous)
    });
    let copy_and_direct_ms = measure(client, warmup, iterations, || {
        let a = into_contiguous(
            client,
            physical_x.clone().binding(),
            f32::elem_type_native(),
        );
        let b = into_contiguous(
            client,
            physical_w.clone().binding(),
            f32::elem_type_native(),
        );
        execute(&a, &b, &out_contiguous);
    });
    let strided_direct_ms = measure(client, warmup, iterations, || {
        execute(&physical_x, &physical_w, &out_strided)
    });
    Record {
        name,
        shape: [cin, cout, len, k],
        dilation,
        iterations,
        reference_max_abs,
        strided_max_abs,
        resident_contiguous_ms,
        copy_and_direct_ms,
        strided_direct_ms,
        predicted_speedup: copy_and_direct_ms / strided_direct_ms,
    }
}

fn check_nd<const N: usize>(client: &Client, spatial: [usize; N]) -> f32 {
    let cin = 3usize;
    let cout = 5usize;
    let kernels = [3usize; N];
    let spatial_volume = spatial.iter().product::<usize>();
    let kernel_volume = kernels.iter().product::<usize>();
    let x: Vec<f32> = (0..cin * spatial_volume)
        .map(|q| (q % 53) as f32 / 19. - 1.1)
        .collect();
    let w: Vec<f32> = (0..cout * cin * kernel_volume)
        .map(|q| (q % 37) as f32 / 41. - 0.5)
        .collect();
    let mut input_shape = vec![1];
    input_shape.extend(spatial.iter().copied());
    input_shape.push(cin);
    let mut weight_shape = vec![cout];
    weight_shape.extend(kernels.iter().copied());
    weight_shape.push(cin);
    let mut input_strides = vec![cin * spatial_volume];
    let mut weight_strides = vec![cin * kernel_volume];
    for i in 0..N {
        input_strides.push(spatial[i + 1..].iter().product::<usize>());
        weight_strides.push(kernels[i + 1..].iter().product::<usize>());
    }
    input_strides.push(spatial_volume);
    weight_strides.push(kernel_volume);
    let input = TensorHandle::new(
        client.create_from_slice(bytemuck::cast_slice(&x)),
        input_shape.clone(),
        input_strides,
        f32::elem_type_native(),
    );
    let weight = TensorHandle::new(
        client.create_from_slice(bytemuck::cast_slice(&w)),
        weight_shape,
        weight_strides,
        f32::elem_type_native(),
    );
    let input_contig = into_contiguous(client, input.clone().binding(), f32::elem_type_native());
    let weight_contig = into_contiguous(client, weight.clone().binding(), f32::elem_type_native());
    let mut output_shape = vec![1];
    output_shape.extend(spatial);
    output_shape.push(cout);
    let out_a = TensorHandle::empty(client, output_shape.clone(), f32::elem_type_native());
    let out_b = TensorHandle::empty(client, output_shape, f32::elem_type_native());
    let opts = ConvolutionArgs {
        stride: [1; N],
        padding: [1; N],
        dilation: [1; N],
    };
    for (x, w, y) in [
        (&input_contig, &weight_contig, &out_a),
        (&input, &weight, &out_b),
    ] {
        launch_direct::<N>(
            client,
            DirectTensors {
                input: x.clone().binding(),
                weight: w.clone().binding(),
                bias: None,
                out: y.clone().binding(),
            },
            opts.clone(),
            1,
            f32::elem_type_native(),
        )
        .unwrap();
    }
    let a = read(client, &out_a);
    let b = read(client, &out_b);
    let err = a
        .iter()
        .zip(&b)
        .map(|(u, v)| (u - v).abs())
        .fold(0f32, f32::max);
    assert!(err <= 2e-5, "ND convolution differs: rank={N} error={err}");
    err
}

fn check_groups(client: &Client) -> f32 {
    let cin = 4usize;
    let cout = 4usize;
    let per_group = 2usize;
    let length = 11usize;
    let kernel = 3usize;
    let input: Vec<f32> = (0..cin * length)
        .map(|i| (i % 17) as f32 / 5. - 1.)
        .collect();
    let weight: Vec<f32> = (0..cout * per_group * kernel)
        .map(|i| (i % 19) as f32 / 7. - 1.)
        .collect();
    let x = tensor(client, &input, [1, length, cin], [cin * length, 1, length]);
    let w = tensor(
        client,
        &weight,
        [cout, kernel, per_group],
        [per_group * kernel, 1, kernel],
    );
    let xc = into_contiguous(client, x.clone().binding(), f32::elem_type_native());
    let wc = into_contiguous(client, w.clone().binding(), f32::elem_type_native());
    let out_c = TensorHandle::empty(client, [1, length, cout], f32::elem_type_native());
    let out_s = TensorHandle::empty(client, [1, length, cout], f32::elem_type_native());
    for (x, w, y) in [(&xc, &wc, &out_c), (&x, &w, &out_s)] {
        launch_direct::<1>(
            client,
            DirectTensors {
                input: x.clone().binding(),
                weight: w.clone().binding(),
                bias: None,
                out: y.clone().binding(),
            },
            ConvolutionArgs {
                stride: [1],
                padding: [1],
                dilation: [1],
            },
            2,
            f32::elem_type_native(),
        )
        .unwrap();
    }
    let baseline = read(client, &out_c);
    let candidate = read(client, &out_s);
    let error = baseline
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        error <= 2e-5,
        "grouped strided convolution differs: {error}"
    );
    error
}

fn main() {
    let client = CpuRuntime::client(&Default::default());
    let warmup = 5;
    let iterations = 12;
    let shapes = [
        ("small K3", 4, 8, 15, 3, 1),
        ("small K11 D3", 4, 8, 63, 11, 3),
        ("NSF K3", 256, 256, 240, 3, 1),
        ("NSF K11 D5", 128, 128, 2400, 11, 5),
        ("NSF K11", 32, 32, 9600, 11, 1),
    ];
    let results = shapes
        .into_iter()
        .map(|(name, cin, cout, len, k, d)| {
            eprintln!("running {name}");
            case(&client, name, cin, cout, len, k, d, warmup, iterations)
        })
        .collect::<Vec<_>>();
    eprintln!(
        "2D strided parity max_abs={}",
        check_nd::<2>(&client, [4, 5])
    );
    eprintln!(
        "3D strided parity max_abs={}",
        check_nd::<3>(&client, [3, 4, 5])
    );
    eprintln!("grouped Conv1d parity max_abs={}", check_groups(&client));
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}
