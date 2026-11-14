use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubecl_cpu::CpuRuntime;
use cubecl_server::runtime::Runtime;
use cubek_convolution::kernels::forward::{
    launch_pack_conv1d_weights, launch_prepacked_temporal_quad_interior,
};
use cubek_convolution::{ConvolutionArgs, DirectTensors, launch_direct};
use serde::Serialize;
use std::time::Instant;

#[cube(launch_unchecked, address_type = "dynamic")]
fn copy_kernel<E: Numeric>(
    x: &Tensor<E>,
    y: &mut Tensor<E>,
    n: u32,
    #[define(E)] _dtype: ElemType,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        y[i as usize] = x[i as usize];
    }
}
#[cube(launch_unchecked, address_type = "dynamic")]
fn add_kernel<E: Numeric>(x: &Tensor<E>, y: &mut Tensor<E>, n: u32, #[define(E)] _dtype: ElemType) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        y[i as usize] = y[i as usize] + x[i as usize];
    }
}
#[cube(launch_unchecked, address_type = "dynamic")]
fn div_kernel<E: Numeric>(x: &Tensor<E>, y: &mut Tensor<E>, n: u32, #[define(E)] _dtype: ElemType) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        y[i as usize] = x[i as usize] / E::from_int(3);
    }
}
#[cube(launch_unchecked, address_type = "dynamic")]
fn fused_mean_kernel<E: Numeric>(
    a: &Tensor<E>,
    b: &Tensor<E>,
    c: &Tensor<E>,
    y: &mut Tensor<E>,
    n: u32,
    #[define(E)] _dtype: ElemType,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        y[i as usize] = ((a[i as usize] + b[i as usize]) + c[i as usize]) / E::from_int(3);
    }
}
#[cube(launch_unchecked, address_type = "dynamic")]
fn ncl_to_nlc_kernel<E: Numeric>(
    x: &Tensor<E>,
    y: &mut Tensor<E>,
    len: u32,
    channels: u32,
    n: u32,
    #[define(E)] _dtype: ElemType,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let c = i % channels;
        let t = (i / channels) % len;
        let b = i / (channels * len);
        y[i as usize] = x[(b * len * channels + c * len + t) as usize];
    }
}
#[cube(launch_unchecked, address_type = "dynamic")]
fn nlc_to_ncl_kernel<E: Numeric>(
    x: &Tensor<E>,
    y: &mut Tensor<E>,
    len: u32,
    channels: u32,
    n: u32,
    #[define(E)] _dtype: ElemType,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let t = i % len;
        let c = (i / len) % channels;
        let b = i / (channels * len);
        y[i as usize] = x[(b * len * channels + t * channels + c) as usize];
    }
}
fn tensor(client: &Client, v: &[f32], shape: impl Into<Shape>) -> TensorHandle {
    TensorHandle::new_contiguous(
        shape,
        client.create_from_slice(bytemuck::cast_slice(v)),
        f32::elem_type_native(),
    )
}
fn empty(client: &Client, shape: impl Into<Shape>) -> TensorHandle {
    TensorHandle::empty(client, shape, f32::elem_type_native())
}
fn read(client: &Client, t: &TensorHandle) -> Vec<f32> {
    let bytes = client.read_one_unchecked(t.clone().handle);
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}
fn launch_dim(client: &Client, n: usize) -> (CubeDim, CubeCount) {
    let cube_dim = CubeDim::new(client, n);
    (cube_dim, calculate_cube_count_elemwise(client, n, cube_dim))
}
fn launch_serial(
    client: &Client,
    a: &TensorHandle,
    b: &TensorHandle,
    c: &TensorHandle,
    tmp: &TensorHandle,
    dst: &TensorHandle,
    n: usize,
) {
    let (d, g) = launch_dim(client, n);
    let dtype = f32::elem_type_native();
    unsafe {
        copy_kernel::launch_unchecked(
            client,
            g.clone(),
            d,
            AddressType::U32,
            a.clone().binding().into_tensor_arg(),
            tmp.clone().binding().into_tensor_arg(),
            n as u32,
            dtype,
        );
        add_kernel::launch_unchecked(
            client,
            g.clone(),
            d,
            AddressType::U32,
            b.clone().binding().into_tensor_arg(),
            tmp.clone().binding().into_tensor_arg(),
            n as u32,
            dtype,
        );
        add_kernel::launch_unchecked(
            client,
            g.clone(),
            d,
            AddressType::U32,
            c.clone().binding().into_tensor_arg(),
            tmp.clone().binding().into_tensor_arg(),
            n as u32,
            dtype,
        );
        div_kernel::launch_unchecked(
            client,
            g.clone(),
            d,
            AddressType::U32,
            tmp.clone().binding().into_tensor_arg(),
            dst.clone().binding().into_tensor_arg(),
            n as u32,
            dtype,
        );
    }
}
fn launch_fused(
    client: &Client,
    a: &TensorHandle,
    b: &TensorHandle,
    c: &TensorHandle,
    dst: &TensorHandle,
    n: usize,
) {
    let (d, g) = launch_dim(client, n);
    unsafe {
        fused_mean_kernel::launch_unchecked(
            client,
            g,
            d,
            AddressType::U32,
            a.clone().binding().into_tensor_arg(),
            b.clone().binding().into_tensor_arg(),
            c.clone().binding().into_tensor_arg(),
            dst.clone().binding().into_tensor_arg(),
            n as u32,
            f32::elem_type_native(),
        );
    }
}
fn launch_layout(
    client: &Client,
    x: &TensorHandle,
    y: &TensorHandle,
    len: usize,
    ch: usize,
    batch: usize,
    forward: bool,
) {
    let (d, g) = launch_dim(client, len * ch * batch);
    let (x, y) = (
        x.clone().binding().into_tensor_arg(),
        y.clone().binding().into_tensor_arg(),
    );
    let dtype = f32::elem_type_native();
    unsafe {
        if forward {
            ncl_to_nlc_kernel::launch_unchecked(
                client,
                g,
                d,
                AddressType::U32,
                x,
                y,
                len as u32,
                ch as u32,
                (len * ch * batch) as u32,
                dtype,
            );
        } else {
            nlc_to_ncl_kernel::launch_unchecked(
                client,
                g,
                d,
                AddressType::U32,
                x,
                y,
                len as u32,
                ch as u32,
                (len * ch * batch) as u32,
                dtype,
            );
        }
    }
}
fn sync(client: &Client) {
    futures_lite::future::block_on(client.sync()).expect("CPU sync");
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn measure(client: &Client, warmup: usize, iterations: usize, mut op: impl FnMut()) -> f64 {
    let mut samples = Vec::with_capacity(iterations);
    for i in 0..warmup + iterations {
        let start = Instant::now();
        op();
        sync(client);
        if i >= warmup {
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    median(&mut samples)
}
#[derive(Serialize)]
struct Case {
    name: &'static str,
    length: usize,
    channels: usize,
    serial_ms: f64,
    fused_ms: f64,
    speedup: f64,
    layout_roundtrip_ms: f64,
    layout_forward_ms: f64,
    layout_reverse_ms: f64,
    max_abs_error: f32,
    exact_layout: bool,
}
#[derive(Serialize)]
struct Report {
    backend: &'static str,
    warmup: usize,
    iterations: usize,
    cases: Vec<Case>,
    conv_cases: Vec<ConvCase>,
}
fn case(
    client: &Client,
    name: &'static str,
    len: usize,
    ch: usize,
    warmup: usize,
    iterations: usize,
) -> Case {
    let n = len * ch;
    let values = |seed: usize| {
        (0..n)
            .map(|i| ((i * 37 + seed * 71) % 601) as f32 / 257.0 - 1.0)
            .collect::<Vec<_>>()
    };
    let a = tensor(client, &values(1), Shape::new([1, len, ch]));
    let b = tensor(client, &values(2), Shape::new([1, len, ch]));
    let c = tensor(client, &values(3), Shape::new([1, len, ch]));
    let tmp = empty(client, Shape::new([1, len, ch]));
    let out0 = empty(client, Shape::new([1, len, ch]));
    let out1 = empty(client, Shape::new([1, len, ch]));
    let serial_ms = measure(client, warmup, iterations, || {
        launch_serial(client, &a, &b, &c, &tmp, &out0, n)
    });
    let fused_ms = measure(client, warmup, iterations, || {
        launch_fused(client, &a, &b, &c, &out1, n)
    });
    let max_abs_error = read(client, &out0)
        .iter()
        .zip(read(client, &out1).iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let source_values = values(4);
    let source = tensor(client, &source_values, Shape::new([1, ch, len]));
    let nlc = empty(client, Shape::new([1, len, ch]));
    let back = empty(client, Shape::new([1, ch, len]));
    let layout_forward_ms = measure(client, warmup, iterations, || {
        launch_layout(client, &source, &nlc, len, ch, 1, true)
    });
    let layout_reverse_ms = measure(client, warmup, iterations, || {
        launch_layout(client, &nlc, &back, len, ch, 1, false)
    });
    let layout_roundtrip_ms = measure(client, warmup, iterations, || {
        launch_layout(client, &source, &nlc, len, ch, 1, true);
        launch_layout(client, &nlc, &back, len, ch, 1, false);
    });
    Case {
        name,
        length: len,
        channels: ch,
        serial_ms,
        fused_ms,
        speedup: serial_ms / fused_ms,
        layout_roundtrip_ms,
        layout_forward_ms,
        layout_reverse_ms,
        max_abs_error,
        exact_layout: read(client, &back) == source_values,
    }
}

#[derive(Serialize)]
struct ConvCase {
    name: &'static str,
    cin: usize,
    cout: usize,
    length: usize,
    kernel: usize,
    dilation: usize,
    direct_materialized_ms: f64,
    resident_direct_ms: f64,
    explicit_layout_and_quad_ms: f64,
    resident_quad_ms: f64,
    materialized_direct_vs_quad_pipeline: f64,
    pack_direct_from_view_ms: f64,
    pack_copy_then_pack_ms: f64,
    packing_exact: bool,
    max_abs_error: f32,
}
fn run_conv_case(
    client: &Client,
    name: &'static str,
    cin: usize,
    cout: usize,
    len: usize,
    k: usize,
    dilation: usize,
    warmup: usize,
    iterations: usize,
) -> ConvCase {
    let xdata = (0..cin * len)
        .map(|i| ((i * 29 % 311) as f32 - 150.0) / 211.0)
        .collect::<Vec<_>>();
    let wdata = (0..cout * cin * k)
        .map(|i| ((i * 13 % 199) as f32 - 100.0) / 379.0)
        .collect::<Vec<_>>();
    let xncl = tensor(client, &xdata, Shape::new([1, cin, len]));
    let wncl = tensor(client, &wdata, Shape::new([cout, cin, k]));
    let xnlc = empty(client, Shape::new([1, len, cin]));
    launch_layout(client, &xncl, &xnlc, len, cin, 1, true);
    let wnlc = empty(client, Shape::new([cout, k, cin]));
    launch_layout(client, &wncl, &wnlc, k, cin, cout, true);
    let mut packed = vec![0.0f32; wdata.len()];
    for kk in 0..k {
        for ci in 0..cin {
            for oc in 0..cout {
                packed[(kk * cin + ci) * cout + oc] = wdata[(oc * cin + ci) * k + kk];
            }
        }
    }
    let dtype = f32::elem_type_native();
    let wp = empty(client, Shape::new([1, k, cin, cout]));
    let wp_copy = empty(client, Shape::new([1, k, cin, cout]));
    let wview = TensorHandle::new(
        wncl.handle.clone(),
        Shape::new([cout, k, cin]),
        [cin * k, 1, k],
        dtype,
    );
    let pack_direct_from_view_ms = measure(client, warmup, iterations, || {
        launch_pack_conv1d_weights(client, wview.clone().binding(), wp.clone().binding(), dtype)
            .expect("pack strided weight");
    });
    let pack_copy_then_pack_ms = measure(client, warmup, iterations, || {
        launch_layout(client, &wncl, &wnlc, k, cin, cout, true);
        launch_pack_conv1d_weights(
            client,
            wnlc.clone().binding(),
            wp_copy.clone().binding(),
            dtype,
        )
        .expect("pack materialized weight");
    });
    let packing_exact = read(client, &wp) == packed && read(client, &wp_copy) == packed;

    let outdirect = empty(client, Shape::new([1, len, cout]));
    let outquad = empty(client, Shape::new([1, len, cout]));
    let opts = ConvolutionArgs {
        stride: [1],
        padding: [(k - 1) * dilation / 2],
        dilation: [dilation],
    };
    let mut direct = || {
        launch_direct::<1>(
            client,
            DirectTensors {
                input: xnlc.clone().binding(),
                weight: wnlc.clone().binding(),
                bias: None,
                out: outdirect.clone().binding(),
            },
            opts.clone(),
            1,
            dtype,
        )
        .expect("direct strided")
    };
    let mut quad = || {
        launch_prepacked_temporal_quad_interior(
            client,
            DirectTensors {
                input: xnlc.clone().binding(),
                weight: wp.clone().binding(),
                bias: None,
                out: outquad.clone().binding(),
            },
            opts.clone(),
            1,
            dtype,
        )
        .expect("quad interior")
    };
    let resident_direct_ms = measure(client, warmup, iterations, &mut direct);
    let direct_materialized_ms = measure(client, warmup, iterations, || {
        launch_layout(client, &xncl, &xnlc, len, cin, 1, true);
        launch_layout(client, &wncl, &wnlc, k, cin, cout, true);
        direct();
    });
    let resident_quad_ms = measure(client, warmup, iterations, &mut quad);
    let explicit_layout_and_quad_ms = measure(client, warmup, iterations, || {
        launch_layout(client, &xncl, &xnlc, len, cin, 1, true);
        quad();
    });
    let max_abs_error = read(client, &outdirect)
        .iter()
        .zip(read(client, &outquad).iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    ConvCase {
        name,
        cin,
        cout,
        length: len,
        kernel: k,
        dilation,
        direct_materialized_ms,
        resident_direct_ms,
        explicit_layout_and_quad_ms,
        resident_quad_ms,
        materialized_direct_vs_quad_pipeline: direct_materialized_ms / explicit_layout_and_quad_ms,
        pack_direct_from_view_ms,
        pack_copy_then_pack_ms,
        packing_exact,
        max_abs_error,
    }
}
fn main() {
    let mut args = std::env::args().skip(1);
    let mut warmup = 8usize;
    let mut iterations = 30usize;
    let mut output = None::<String>;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--warmup" => warmup = args.next().unwrap().parse().unwrap(),
            "--iterations" => iterations = args.next().unwrap().parse().unwrap(),
            "--json" => output = Some(args.next().unwrap()),
            _ => panic!("Unknown {arg}"),
        }
    }
    assert!((1..=1000).contains(&iterations));
    let client = CpuRuntime::client(&Default::default());
    let cases = [
        ("NSF K3", 240, 256),
        ("NSF K11 D5", 2400, 128),
        ("NSF K11", 9600, 32),
    ]
    .into_iter()
    .map(|(name, len, ch)| {
        eprintln!("[{name}] residual epilogue & layout");
        case(&client, name, len, ch, warmup, iterations)
    })
    .collect();
    let conv_cases=[("NSF K3",256,256,240,3,1),("NSF K11 D5",128,128,2400,11,5),("NSF K11",32,32,9600,11,1)].into_iter().map(|(name,cin,cout,len,k,dilation)|{
        eprintln!("[{name}] comparing Burn physical materialization + Direct with materialization + packed QuadInterior");
        run_conv_case(&client,name,cin,cout,len,k,dilation,warmup,iterations)
    }).collect();
    let report = Report {
        backend: "cpu",
        warmup,
        iterations,
        cases,
        conv_cases,
    };
    let json = serde_json::to_string_pretty(&report).unwrap();
    println!("{json}");
    if let Some(path) = output {
        std::fs::write(path, json).unwrap();
    }
    assert!(
        report
            .cases
            .iter()
            .all(|x| x.max_abs_error <= 1e-5 && x.exact_layout)
    );
    assert!(
        report
            .conv_cases
            .iter()
            .all(|x| x.max_abs_error <= 2e-4 && x.packing_exact)
    );
}
