use burn::tensor::{TensorData, ops::ConvOptions};
use burn_flex::{
    FlexTensor,
    ops::conv::{conv1d_f16, conv1d_f32, conv1d_f64},
};
use half::f16;

fn tensor<T: burn::tensor::Element>(data: Vec<T>, shape: Vec<usize>) -> FlexTensor {
    FlexTensor::from_data(TensorData::new(data, shape))
}

#[test]
fn conv1d_grouped_padding_stride_dilation() {
    let (b, ci, co, groups, n, k) = (2, 8, 12, 2, 53, 5);
    let (stride, pad, dilation) = (2, 4, 2);
    let x: Vec<f32> = (0..b * ci * n)
        .map(|j| ((j * 13 % 101) as f32 - 50.) / 37.)
        .collect();
    let w: Vec<f32> = (0..co * (ci / groups) * k)
        .map(|j| ((j * 7 % 79) as f32 - 40.) / 97.)
        .collect();
    let bias: Vec<f32> = (0..co).map(|i| i as f32 / 30.).collect();
    let opts = ConvOptions::new([stride], [pad], [dilation], groups);
    let y = conv1d_f32(
        tensor(x.clone(), vec![b, ci, n]),
        tensor(w.clone(), vec![co, ci / groups, k]),
        Some(tensor(bias.clone(), vec![co])),
        &opts,
    );
    let y: Vec<f32> = y.into_data().try_into_vec().unwrap();
    let nout = (n + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    assert_eq!(y.len(), b * co * nout);
    for batch in 0..b {
        for c in 0..co {
            for o in 0..nout {
                let mut ref_value = bias[c];
                for ic in 0..ci / groups {
                    for ki in 0..k {
                        let p = o as isize * stride as isize + ki as isize * dilation as isize
                            - pad as isize;
                        if (0..n as isize).contains(&p) {
                            let in_channel = c / (co / groups) * (ci / groups) + ic;
                            ref_value += w[(c * (ci / groups) + ic) * k + ki]
                                * x[(batch * ci + in_channel) * n + p as usize];
                        }
                    }
                }
                let actual = y[(batch * co + c) * nout + o];
                assert!(
                    (actual - ref_value).abs() < 1e-3,
                    "{batch} {c} {o}: {actual} != {ref_value}"
                );
            }
        }
    }
}

#[test]
fn conv1d_depthwise_nan_padding_propagation() {
    let opts = ConvOptions::new([1], [1], [1], 2);
    let y = conv1d_f32(
        tensor(vec![1f32, 2., 3., 3., 2., 1.], vec![1, 2, 3]),
        tensor(vec![f32::INFINITY, 1., 1., 1., 1., 1.], vec![2, 1, 3]),
        None,
        &opts,
    );
    let y: Vec<f32> = y.into_data().try_into_vec().unwrap();
    assert!(y[0].is_nan());
    assert_eq!(&y[3..6], &[5., 6., 3.]);
}

#[test]
fn conv1d_grouped_nan_padding_propagation() {
    let opts = ConvOptions::new([1], [1], [1], 1);
    let y = conv1d_f32(
        tensor(vec![1f32, 2., 3.], vec![1, 1, 3]),
        tensor(vec![f32::INFINITY, 1., 1.], vec![1, 1, 3]),
        None,
        &opts,
    );
    let y: Vec<f32> = y.into_data().try_into_vec().unwrap();
    assert!(y[0].is_nan());
    assert!(y[1].is_infinite());
}

#[test]
fn conv1d_f16_accumulates_before_cast() {
    let x = vec![f16::from_f32(128.); 10];
    let w: Vec<f16> = (0..5)
        .flat_map(|_| [f16::from_f32(128.), f16::from_f32(-128.)])
        .collect();
    let y = conv1d_f16(
        tensor(x, vec![1, 5, 2]),
        tensor(w, vec![1, 5, 2]),
        None,
        &ConvOptions::new([1], [0], [1], 1),
    );
    let y: Vec<f16> = y.into_data().try_into_vec().unwrap();
    assert_eq!(y.len(), 1);
    assert_eq!(y[0].to_f32(), 0.);
}

#[test]
fn conv1d_zero_channels_returns_bias_without_pointer_arithmetic() {
    let y = conv1d_f32(
        tensor(Vec::<f32>::new(), vec![1, 0, 4]),
        tensor(Vec::<f32>::new(), vec![1, 0, 3]),
        Some(tensor(vec![4.25f32], vec![1])),
        &ConvOptions::new([1], [0], [1], 1),
    );
    let values: Vec<f32> = y.into_data().try_into_vec().unwrap();
    assert_eq!(values, vec![4.25, 4.25]);
}

#[test]
fn conv1d_f64_parity_simple() {
    let y = conv1d_f64(
        tensor(vec![1., 2., 3., 4.], vec![1, 1, 4]),
        tensor(vec![0.5, 1.], vec![1, 1, 2]),
        None,
        &ConvOptions::new([1], [0], [1], 1),
    );
    let values: Vec<f64> = y.into_data().try_into_vec().unwrap();
    assert_eq!(values, vec![2.5, 4., 5.5]);
}

#[test]
fn conv1d_randomized_grouped_depthwise_and_dilated_reference() {
    for case in 0usize..80 {
        let groups = [1, 2, 4, 8][case % 4];
        let batch = 1 + case % 2;
        let cin_group = 1 + (case * 3 % 6);
        let cout_group = if case % 7 == 0 { 1 } else { 1 + case * 5 % 7 };
        let ci = cin_group * groups;
        let co = cout_group * groups;
        let n = 15 + (case * 11 % 45);
        let k = 1 + (case * 13 % 6);
        let stride = 1 + case % 3;
        let dilation = 1 + case * 2 % 4;
        let pad = case * 7 % 6;
        if n + 2 * pad < dilation * (k - 1) + 1 {
            continue;
        }
        let x: Vec<f32> = (0..batch * ci * n)
            .map(|j| ((j * 19 + case * 3) % 79) as f32 / 51. - 0.7)
            .collect();
        let w: Vec<f32> = (0..co * cin_group * k)
            .map(|j| ((j * 23 + case * 7) % 97) as f32 / 82. - 0.4)
            .collect();
        let opts = ConvOptions::new([stride], [pad], [dilation], groups);
        let actual = conv1d_f32(
            tensor(x.clone(), vec![batch, ci, n]),
            tensor(w.clone(), vec![co, cin_group, k]),
            None,
            &opts,
        );
        let actual: Vec<f32> = actual.into_data().try_into_vec().unwrap();
        let out_n = (n + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        assert_eq!(actual.len(), batch * co * out_n);
        for b in 0..batch {
            for c in 0..co {
                for pos in 0..out_n {
                    let mut expected = 0f32;
                    for ic in 0..cin_group {
                        for ki in 0..k {
                            let x_pos = pos as isize * stride as isize
                                + ki as isize * dilation as isize
                                - pad as isize;
                            if (0..n as isize).contains(&x_pos) {
                                let input_channel = c / cout_group * cin_group + ic;
                                expected += x[(b * ci + input_channel) * n + x_pos as usize]
                                    * w[(c * cin_group + ic) * k + ki];
                            }
                        }
                    }
                    let got = actual[(b * co + c) * out_n + pos];
                    assert!(
                        (got - expected).abs() < 2e-4,
                        "case {case} b={b} c={c} pos={pos}: {got} != {expected}"
                    );
                }
            }
        }
    }
}
