use cubecl::{
    calculate_cube_count_elemwise, client::Client, prelude::*, std::tensor::TensorHandle,
};
use cubecl_cpu::CpuRuntime;
use cubecl_server::runtime::Runtime;
use cubek_convolution::{ConvolutionArgs, DirectTensors, launch_direct};
use serde::Serialize;
use std::hint::black_box;
use std::time::Instant;

#[cube(launch_unchecked, address_type = "dynamic")]
fn leaky_kernel<E: Numeric>(
    input: &Tensor<E>,
    out: &mut Tensor<E>,
    elements: u32,
    #[define(E)] _dtype: ElemType,
) {
    let i = ABSOLUTE_POS as u32;
    if i < elements {
        let value = input[i as usize];
        out[i as usize] = if value < E::from_int(0) {
            value / E::from_int(10)
        } else {
            value
        };
    }
}

#[derive(Serialize)]
struct Record {
    name: &'static str,
    frames: usize,
    cin: usize,
    cout: usize,
    kernel: usize,
    dilation: usize,
    baseline_materialize_ms: f64,
    resident_layout_ms: f64,
    speedup: f64,
    max_abs_error: f32,
}

fn tensor(client: &Client, data: &[f32], shape: [usize; 3], strides: [usize; 3]) -> TensorHandle {
    TensorHandle::new(
        client.create_from_slice(bytemuck::cast_slice(data)),
        shape,
        strides,
        f32::elem_type_native(),
    )
}

fn out(client: &Client, shape: [usize; 3]) -> TensorHandle {
    TensorHandle::new_contiguous(
        shape,
        client.empty(shape.iter().product::<usize>() * 4),
        f32::elem_type_native(),
    )
}

fn contiguous(client: &Client, input: TensorHandle) -> TensorHandle {
    cubecl::std::tensor::into_contiguous(client, input.binding(), f32::elem_type_native())
}

fn leaky(client: &Client, input: &TensorHandle, output: &TensorHandle, n: usize) {
    let dim = CubeDim::new(client, n);
    let count = calculate_cube_count_elemwise(client, n, dim);
    unsafe {
        leaky_kernel::launch_unchecked(
            client,
            count,
            dim,
            AddressType::U32,
            input.clone().binding().into_tensor_arg(),
            output.clone().binding().into_tensor_arg(),
            n as u32,
            f32::elem_type_native(),
        );
    }
}

fn conv(
    client: &Client,
    x: &TensorHandle,
    w: &TensorHandle,
    y: &TensorHandle,
    kernel: usize,
    dilation: usize,
) {
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
            dilation: [dilation],
            padding: [(kernel - 1) * dilation / 2],
        },
        1,
        f32::elem_type_native(),
    )
    .unwrap();
}

fn execute(
    client: &Client,
    input_ncl: &TensorHandle,
    weight_nck: &TensorHandle,
    weight_prepacked: &TensorHandle,
    len: usize,
    channels: usize,
    kernel: usize,
    dilation: usize,
    resident: bool,
) -> TensorHandle {
    let xnlc = contiguous(client, input_ncl.clone());
    let w = if resident {
        weight_prepacked.clone()
    } else {
        contiguous(client, weight_nck.clone())
    };
    let y1 = out(client, [1, len, channels]);
    conv(client, &xnlc, &w, &y1, kernel, dilation);

    let yact = if resident {
        let dst = out(client, [1, len, channels]);
        leaky(client, &y1, &dst, len * channels);
        dst
    } else {
        let view_ncl = TensorHandle::new(
            y1.handle.clone(),
            [1, channels, len],
            [len * channels, 1, channels],
            f32::elem_type_native(),
        );
        let physical_ncl = contiguous(client, view_ncl);
        let activated_ncl = out(client, [1, channels, len]);
        leaky(client, &physical_ncl, &activated_ncl, len * channels);
        let next_view_nlc = TensorHandle::new(
            activated_ncl.handle.clone(),
            [1, len, channels],
            [len * channels, 1, len],
            f32::elem_type_native(),
        );
        contiguous(client, next_view_nlc)
    };
    let w2 = if resident {
        weight_prepacked.clone()
    } else {
        contiguous(client, weight_nck.clone())
    };
    let y2 = out(client, [1, len, channels]);
    conv(client, &yact, &w2, &y2, kernel, dilation);
    y2
}

fn fetch(client: &Client, handle: TensorHandle) -> Vec<f32> {
    black_box(bytemuck::cast_slice::<u8, f32>(&client.read_one_unchecked(handle.handle)).to_vec())
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn measure(
    client: &Client,
    warmup: usize,
    iterations: usize,
    mut run: impl FnMut() -> TensorHandle,
) -> f64 {
    let mut values = Vec::new();
    for i in 0..(warmup + iterations) {
        let started = Instant::now();
        let _ = fetch(client, run());
        if i >= warmup {
            values.push(started.elapsed().as_secs_f64() * 1000.);
        }
    }
    median(values)
}

fn case(
    client: &Client,
    name: &'static str,
    len: usize,
    ch: usize,
    kernel: usize,
    dilation: usize,
) -> Record {
    let xs: Vec<f32> = (0..len * ch)
        .map(|i| ((i * 23 % 229) as f32 - 114.) / 149.)
        .collect();
    let ws: Vec<f32> = (0..ch * ch * kernel)
        .map(|i| ((i * 11 % 193) as f32 - 97.) / 237.)
        .collect();
    let input = tensor(client, &xs, [1, len, ch], [len * ch, 1, len]);
    let weight = tensor(client, &ws, [ch, kernel, ch], [ch * kernel, 1, kernel]);
    let wnlc = contiguous(client, weight.clone());
    let baseline = fetch(
        client,
        execute(
            client, &input, &weight, &wnlc, len, ch, kernel, dilation, false,
        ),
    );
    let candidate = fetch(
        client,
        execute(
            client, &input, &weight, &wnlc, len, ch, kernel, dilation, true,
        ),
    );
    let error = baseline
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(error <= 1e-5, "pipeline parity failed for {name}: {error}");
    let base = measure(client, 4, 12, || {
        execute(
            client, &input, &weight, &wnlc, len, ch, kernel, dilation, false,
        )
    });
    let optimized = measure(client, 4, 12, || {
        execute(
            client, &input, &weight, &wnlc, len, ch, kernel, dilation, true,
        )
    });
    Record {
        name,
        frames: len,
        cin: ch,
        cout: ch,
        kernel,
        dilation,
        baseline_materialize_ms: base,
        resident_layout_ms: optimized,
        speedup: base / optimized,
        max_abs_error: error,
    }
}

fn main() {
    let client = CpuRuntime::client(&Default::default());
    let records = [
        ("small K3", 64, 8, 3, 1),
        ("NSF K3", 240, 256, 3, 1),
        ("NSF K11 D5", 2400, 128, 11, 5),
        ("NSF K11", 9600, 32, 11, 1),
    ]
    .into_iter()
    .map(|(n, t, c, k, d)| {
        eprintln!("{n}");
        case(&client, n, t, c, k, d)
    })
    .collect::<Vec<_>>();
    println!("{}", serde_json::to_string_pretty(&records).unwrap());
}
