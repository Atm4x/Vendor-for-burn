use burn::tensor::{module, ops::ConvOptions, Device, Tensor, TensorData};
use std::time::Instant;

fn reference(x: &[f32], w: &[f32], bias: &[f32], b: usize, ci: usize, co: usize, h: usize, width: usize, kh: usize, kw: usize, stride: [usize; 2], pad: [usize; 2], dilation: [usize; 2], groups: usize) -> (Vec<f32>, usize, usize) {
    let oh = (h + 2 * pad[0] - dilation[0] * (kh - 1) - 1) / stride[0] + 1;
    let ow = (width + 2 * pad[1] - dilation[1] * (kw - 1) - 1) / stride[1] + 1;
    let mut result = vec![0.0; b*co*oh*ow];
    let cin_group = ci / groups;
    let cout_group = co / groups;
    for batch in 0..b { for oc in 0..co { for y in 0..oh { for z in 0..ow {
        let g = oc / cout_group;
        let mut value = bias[oc];
        for ic in 0..cin_group { for ky in 0..kh { for kx in 0..kw {
            let iy = (y*stride[0]+ky*dilation[0]) as isize - pad[0] as isize;
            let ix = (z*stride[1]+kx*dilation[1]) as isize - pad[1] as isize;
            if iy >= 0 && ix >= 0 && iy < h as isize && ix < width as isize {
                let xv = x[((batch*ci+g*cin_group+ic)*h+iy as usize)*width+ix as usize];
                let wv = w[((oc*cin_group+ic)*kh+ky)*kw+kx];
                value += xv*wv;
            }
        }}}
        result[((batch*co+oc)*oh+y)*ow+z] = value;
    }}}}
    (result, oh, ow)
}

fn main() -> Result<(), String> {
    let d=Device::flex();
    let mut checked=0;
    let cases = [(1,4,8,7,8,3,3,[1,1],[1,1],[1,1],1),
      (2,8,8,11,7,3,3,[2,1],[2,0],[2,1],2),
      (2,6,6,9,11,5,1,[1,2],[2,0],[1,2],6),
      (1,3,9,15,13,2,3,[2,2],[0,1],[1,2],3),
      (1,2,4,8,9,1,1,[1,1],[0,0],[1,1],1),
      (1,5,10,8,8,3,3,[1,1],[2,2],[1,1],5)];
    for (b,ci,co,h,ww,kh,kw,s,p,di,g) in cases {
        let x=(0..b*ci*h*ww).map(|i|((i*37%149) as f32-70.0)*0.005).collect::<Vec<_>>();
        let w=(0..co*(ci/g)*kh*kw).map(|i|((i*53%211) as f32-90.0)*0.001).collect::<Vec<_>>();
        let bias=(0..co).map(|i|i as f32*0.01).collect::<Vec<_>>();
        let input:Tensor<4>=Tensor::from_data(TensorData::new(x.clone(),[b,ci,h,ww]),&d);
        let weights:Tensor<4>=Tensor::from_data(TensorData::new(w.clone(),[co,ci/g,kh,kw]),&d);
        let bv:Tensor<1>=Tensor::from_data(TensorData::new(bias.clone(),[co]),&d);
        let got=module::conv2d(input,weights,Some(bv),ConvOptions::new(s,p,di,g)).into_data().try_into_vec::<f32>().map_err(|e|e.to_string())?;
        let (expect,oh,ow)=reference(&x,&w,&bias,b,ci,co,h,ww,kh,kw,s,p,di,g);
        if got.len()!=b*co*oh*ow{return Err(format!("wrong length for group {g}: {} != {}",got.len(),expect.len()));}
        let max=got.iter().zip(expect.iter()).map(|(a,b)|(a-b).abs()).fold(0.0f32, f32::max);
        if max>3e-4 { return Err(format!("parity failed group {g} shape {ci}x{h}x{ww}, max={max}"));}
        checked+=1;
        println!("parity_case={checked} ci={ci} co={co} groups={g} max_abs={max:.8}");
    }
    for (channels,size) in [(16,128),(32,64),(64,32),(128,16),(256,8),(512,4)] {
        let in_values=(0..channels*size*size).map(|i|((i as f32*0.017).sin())*0.1).collect::<Vec<_>>();
        let w_values=(0..channels*channels*9).map(|i|((i as f32*0.003).sin())*0.01).collect::<Vec<_>>();
        let input:Tensor<4>=Tensor::from_data(TensorData::new(in_values,[1,channels,size,size]),&d);
        let weights:Tensor<4>=Tensor::from_data(TensorData::new(w_values,[channels,channels,3,3]),&d);
        for i in 0..8 {
            let started=Instant::now();
            let output=module::conv2d(input.clone(),weights.clone(),None,ConvOptions::new_with_padding([1,1],[(1,1),(1,1)],[1,1],1)).into_data().try_into_vec::<f32>().map_err(|e|e.to_string())?;
            if output.len()!=channels*size*size||!output.iter().all(|v|v.is_finite()){return Err("nonfinite output".into());}
            println!("channels={channels} size={size} pass={i} conv_ms={:.4}",started.elapsed().as_secs_f64()*1000.0);
        }
    }
    Ok(())
}
