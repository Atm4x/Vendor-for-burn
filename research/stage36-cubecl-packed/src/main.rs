use burn_backend::{TensorData, ops::ConvOptions};
use burn_flex::{FlexTensor, ops::conv::conv1d_f32};
use cubecl::{prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubecl_cpu::CpuRuntime;
use cubecl_server::runtime::Runtime;
use cubek_convolution::kernels::forward::{
    launch_pack_conv1d_weights, launch_prepacked_temporal_pair, launch_temporal_pair,
    launch_temporal_pair_vector,
};
use cubek_convolution::{ConvolutionArgs, DirectTensors, launch_direct};
use std::{hint::black_box, time::Instant};

fn tensor(client: &Client, data: &[f32], shape: impl Into<Shape>) -> TensorHandle {
    TensorHandle::new_contiguous(
        shape,
        client.create_from_slice(bytemuck::cast_slice(data)),
        f32::elem_type_native(),
    )
}
fn read(client: &Client, tensor: TensorHandle) -> Vec<f32> {
    let bytes = client.read_one_unchecked(tensor.handle);
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}
fn run_case(
    client: &Client,
    cin: usize,
    cout: usize,
    input_len: usize,
    kernel: usize,
    dilation: usize,
) {
    let pad = (kernel - 1) * dilation / 2;
    let out_len = input_len;
    let input: Vec<f32> = (0..input_len * cin)
        .map(|i| ((i * 29 % 311) as f32 - 150.) / 211.)
        .collect();
    let weights: Vec<f32> = (0..cout * kernel * cin)
        .map(|i| ((i * 13 % 199) as f32 - 100.) / 379.)
        .collect();
    let inp = tensor(client, &input, Shape::new([1, input_len, cin]));
    let wt = tensor(client, &weights, Shape::new([cout, kernel, cin]));
    let mut packed_weights = vec![0f32; weights.len()];
    for k in 0..kernel {
        for c in 0..cin {
            for oc in 0..cout {
                packed_weights[(k * cin + c) * cout + oc] = weights[(oc * kernel + k) * cin + c];
            }
        }
    }
    let packed_wt = TensorHandle::empty(
        client,
        Shape::new([1, kernel, cin, cout]),
        f32::elem_type_native(),
    );
    let src_wt = tensor(client, &weights, Shape::new([cout, kernel, cin]));
    let pack_start = Instant::now();
    launch_pack_conv1d_weights(
        client,
        src_wt.binding(),
        packed_wt.clone().binding(),
        f32::elem_type_native(),
    )
    .unwrap();
    let packed_gpu_data = read(client, packed_wt.clone());
    let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
    let max_pack_diff = packed_gpu_data
        .iter()
        .zip(packed_weights.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    assert_eq!(max_pack_diff, 0.0);
    println!(
        "packed weights: {:.3}ms one-time (includes readback), {} elements",
        pack_ms,
        packed_weights.len()
    );
    let opts = ConvolutionArgs {
        stride: [1],
        padding: [pad],
        dilation: [dilation],
    };
    let dtype = f32::elem_type_native();
    let mut nchw = vec![0f32; input.len()];
    for t in 0..input_len {
        for c in 0..cin {
            nchw[c * input_len + t] = input[t * cin + c];
        }
    }
    let mut nchw_w = vec![0f32; weights.len()];
    for oc in 0..cout {
        for k in 0..kernel {
            for c in 0..cin {
                nchw_w[(oc * cin + c) * kernel + k] = weights[(oc * kernel + k) * cin + c];
            }
        }
    }
    let flex_in = FlexTensor::from_data(TensorData::new(nchw, [1, cin, input_len]));
    let flex_weight = FlexTensor::from_data(TensorData::new(nchw_w, [cout, cin, kernel]));
    let flex_opts = ConvOptions::new([1], [pad], [dilation], 1);
    let mut measurements = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut results = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for round in 0..12 {
        let order = if round % 2 == 0 {
            [0, 1, 4, 3, 2]
        } else {
            [2, 3, 4, 1, 0]
        };
        for variant in order {
            let start = Instant::now();
            let (values, elapsed_ms) = if variant == 4 {
                let result = conv1d_f32(flex_in.clone(), flex_weight.clone(), None, &flex_opts);
                let raw: Vec<f32> = result.into_data().try_into_vec().unwrap();
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                let mut nhwc = vec![0f32; raw.len()];
                for t in 0..out_len {
                    for c in 0..cout {
                        nhwc[t * cout + c] = raw[c * out_len + t];
                    }
                }
                (nhwc, elapsed_ms)
            } else {
                let out = TensorHandle::empty(client, Shape::new([1, out_len, cout]), dtype);
                let tensors = DirectTensors {
                    input: inp.clone().binding(),
                    weight: wt.clone().binding(),
                    bias: None,
                    out: out.clone().binding(),
                };
                if variant == 0 {
                    launch_direct::<1>(client, tensors, opts.clone(), 1, dtype).unwrap();
                } else if variant == 1 {
                    launch_temporal_pair(client, tensors, opts.clone(), 1, dtype).unwrap();
                } else if variant == 2 {
                    launch_temporal_pair_vector(client, tensors, opts.clone(), 1, dtype).unwrap();
                } else {
                    let packed_tensors = DirectTensors {
                        input: inp.clone().binding(),
                        weight: packed_wt.clone().binding(),
                        bias: None,
                        out: out.clone().binding(),
                    };
                    launch_prepacked_temporal_pair(client, packed_tensors, opts.clone(), 1, dtype)
                        .unwrap();
                }
                let values = read(client, out);
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                (values, elapsed_ms)
            };
            black_box(&values);
            if round >= 2 {
                measurements[variant].push(elapsed_ms);
            }
            if round == 2 {
                results[variant] = values;
            }
        }
    }
    let maxdiff = results[0]
        .iter()
        .zip(results[1].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let maxdiff_vector = results[0]
        .iter()
        .zip(results[2].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let maxdiff_packed = results[0]
        .iter()
        .zip(results[3].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let maxdiff_flex = results[0]
        .iter()
        .zip(results[4].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    for m in &mut measurements {
        m.sort_by(|a, b| a.total_cmp(b));
    }
    println!(
        "Conv1d {cin}->{cout}, K={kernel}, dilation={dilation}, T={input_len}: CubeCL direct={:.3}ms, old_pair={:.3}ms, vector_pair={:.3}ms, packed={:.3}ms, Flex={:.3}ms, err_pair={maxdiff:.7}, err_vector={maxdiff_vector:.7}, err_packed={maxdiff_packed:.7}, err_flex={maxdiff_flex:.7}",
        measurements[0][measurements[0].len() / 2],
        measurements[1][measurements[1].len() / 2],
        measurements[2][measurements[2].len() / 2],
        measurements[3][measurements[3].len() / 2],
        measurements[4][measurements[4].len() / 2]
    );
    assert!(
        maxdiff < 0.0002
            && maxdiff_vector < 0.0002
            && maxdiff_packed < 0.0002
            && maxdiff_flex < 0.0002
    );
}
fn main() {
    let client = CpuRuntime::client(&Default::default());
    run_case(&client, 256, 256, 240, 3, 1);
    run_case(&client, 128, 128, 2400, 11, 5);
    run_case(&client, 32, 32, 9600, 11, 1);
    verify_grouped_strided(&client, 2, 3, 2, 2, 19, 3, 2, 3, 2);
    verify_grouped_strided(&client, 1, 1, 1, 8, 17, 5, 1, 4, 1);
    verify_grouped_strided(&client, 2, 4, 4, 2, 21, 3, 3, 2, 1);
}

fn verify_grouped_strided(
    client: &Client,
    batch: usize,
    cin_group: usize,
    cout_group: usize,
    groups: usize,
    in_len: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
) {
    let cin = groups * cin_group;
    let cout = groups * cout_group;
    let out_len = (in_len + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
    let input: Vec<f32> = (0..batch * in_len * cin)
        .map(|i| ((i * 37 % 301) as f32 - 150.) / 177.)
        .collect();
    let original: Vec<f32> = (0..cout * kernel * cin_group)
        .map(|i| ((i * 7 % 233) as f32 - 117.) / 401.)
        .collect();
    let input_t = tensor(client, &input, Shape::new([batch, in_len, cin]));
    let original_t = tensor(client, &original, Shape::new([cout, kernel, cin_group]));
    let packed_t = TensorHandle::empty(
        client,
        Shape::new([groups, kernel, cin_group, cout_group]),
        f32::elem_type_native(),
    );
    launch_pack_conv1d_weights(
        client,
        original_t.clone().binding(),
        packed_t.clone().binding(),
        f32::elem_type_native(),
    )
    .unwrap();
    let opts = ConvolutionArgs {
        stride: [stride],
        padding: [pad],
        dilation: [dilation],
    };
    let output_direct = TensorHandle::empty(
        client,
        Shape::new([batch, out_len, cout]),
        f32::elem_type_native(),
    );
    let output_packed = TensorHandle::empty(
        client,
        Shape::new([batch, out_len, cout]),
        f32::elem_type_native(),
    );
    launch_direct::<1>(
        client,
        DirectTensors {
            input: input_t.clone().binding(),
            weight: original_t.binding(),
            bias: None,
            out: output_direct.clone().binding(),
        },
        opts.clone(),
        groups,
        f32::elem_type_native(),
    )
    .unwrap();
    launch_prepacked_temporal_pair(
        client,
        DirectTensors {
            input: input_t.clone().binding(),
            weight: packed_t.clone().binding(),
            bias: None,
            out: output_packed.clone().binding(),
        },
        opts,
        groups,
        f32::elem_type_native(),
    )
    .unwrap();
    let a = read(client, output_direct);
    let b = read(client, output_packed);
    let diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    println!(
        "grouped check B={batch}, groups={groups}, CinG={cin_group}, CoutG={cout_group}, K={kernel}, S={stride}, D={dilation}, P={pad}, Lin={in_len}, Lout={out_len}: maxdiff={diff:.8}"
    );
    assert!(diff <= 1e-4, "grouped parity failed {diff}");
}
