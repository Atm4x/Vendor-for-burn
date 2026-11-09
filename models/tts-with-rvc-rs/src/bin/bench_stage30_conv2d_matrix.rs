use burn::tensor::{Device, Tensor, TensorData, module, ops::ConvOptions};

fn main() -> Result<(), String> {
    let d = Device::flex();
    for (batch, cin, cout, groups, h, w, kh, kw, pad_h, pad_w, stride, dilation) in [
        (1, 16, 16, 16, 16, 16, 3, 3, 1, 1, 1, 1),
        (2, 6, 4, 2, 20, 17, 3, 3, 2, 2, 2, 2),
        (1, 1, 1, 1, 132, 132, 3, 3, 1, 1, 1, 1),
        (1, 4, 2, 1, 5, 7, 2, 3, 3, 2, 1, 2),
        (2, 12, 6, 3, 18, 13, 3, 3, 1, 1, 1, 1),
        (1, 4, 8, 2, 7, 9, 1, 1, 0, 0, 1, 1),
    ] {
        let data = (0..batch * cin * h * w)
            .map(|i| ((i * 37 % 311) as f32 - 160.0) * 0.002)
            .collect::<Vec<_>>();
        let wt = (0..cout * (cin / groups) * kh * kw)
            .map(|i| ((i * 29 % 251) as f32 - 130.0) * 0.003)
            .collect::<Vec<_>>();
        let x4: Tensor<4> = Tensor::from_data(
            TensorData::new(data.clone(), [batch, cin, h, w]), &d);
        let x5: Tensor<5> = Tensor::from_data(
            TensorData::new(data, [batch, cin, 1, h, w]), &d);
        let w4: Tensor<4> = Tensor::from_data(
            TensorData::new(wt.clone(), [cout, cin / groups, kh, kw]), &d);
        let w5: Tensor<5> = Tensor::from_data(
            TensorData::new(wt, [cout, cin / groups, 1, kh, kw]), &d);
        let a: Vec<f32> = module::conv2d(x4, w4, None,
            ConvOptions::new([stride, stride], [pad_h, pad_w], [dilation, dilation], groups))
            .into_data().try_into_vec().map_err(|e| e.to_string())?;
        let b: Vec<f32> = module::conv3d(x5, w5, None,
            ConvOptions::new([1, stride, stride], [0, pad_h, pad_w], [1, dilation, dilation], groups))
            .into_data().try_into_vec().map_err(|e| e.to_string())?;
        if a.len() != b.len() { return Err(format!("shape mismatch: {cin} {cout}")); }
        let err = a.iter().zip(&b)
            .map(|(&x,&y)| (x-y).abs())
            .fold(0.0f32, f32::max);
        if !err.is_finite() || err > 0.0005 {
            return Err(format!("wrong output: cin={cin} cout={cout} groups={groups} error={err}"));
        }
        println!("PASS cin={cin} cout={cout} groups={groups} HxW={h}x{w} stride={stride} dilation={dilation} out={} max_abs={err:.9}",a.len());
    }
    {
        let data = vec![0.0f32,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0];
        let wt = vec![f32::INFINITY,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0];
        let x4: Tensor<4> = Tensor::from_data(TensorData::new(data.clone(),[1,1,3,3]),&d);
        let x5: Tensor<5> = Tensor::from_data(TensorData::new(data,[1,1,1,3,3]),&d);
        let w4: Tensor<4> = Tensor::from_data(TensorData::new(wt.clone(),[1,1,3,3]),&d);
        let w5: Tensor<5> = Tensor::from_data(TensorData::new(wt,[1,1,1,3,3]),&d);
        let a:Vec<f32> = module::conv2d(x4,w4,None,ConvOptions::new([1,1],[1,1],[1,1],1))
            .into_data().try_into_vec().map_err(|e|e.to_string())?;
        let b:Vec<f32> = module::conv3d(x5,w5,None,ConvOptions::new([1,1,1],[0,1,1],[1,1,1],1))
            .into_data().try_into_vec().map_err(|e|e.to_string())?;
        if a.len()!=b.len() || a.iter().zip(&b).any(|(&x,&y)|
            !(x.is_nan()&&y.is_nan()) && !(x==y || (x-y).abs()<1e-5)) {
            return Err(format!("nonfinite padding mismatch: {a:?} vs {b:?}"));
        }
        println!("PASS nonfinite_padding nan_count={}",a.iter().filter(|v|v.is_nan()).count());
    }
    Ok(())
}
