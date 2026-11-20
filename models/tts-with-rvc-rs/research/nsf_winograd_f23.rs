use std::{hint::black_box, time::Instant};

pub struct F23Plan {
    channels_in: usize,
    channels_out: usize,
    transformed: [Vec<f32>; 4],
}

impl F23Plan {
    pub fn from_oik3(weights: &[f32], channels_in: usize, channels_out: usize) -> Result<Self, String> {
        let expected = channels_in.checked_mul(channels_out).and_then(|n| n.checked_mul(3)).ok_or("weight shape overflow")?;
        if weights.len() != expected || weights.iter().any(|v| !v.is_finite()) {
            return Err("F23 supports only finite f32 OIK3 weights".to_string());
        }
        let len = channels_in * channels_out;
        let mut transformed = std::array::from_fn(|_| vec![0f32; len]);
        for oc in 0..channels_out {
            for ic in 0..channels_in {
                let i = (oc * channels_in + ic) * 3;
                let a = weights[i];
                let b = weights[i + 1];
                let c = weights[i + 2];
                let dst = oc * channels_in + ic;
                transformed[0][dst] = a;
                transformed[1][dst] = (a + b + c) * 0.5;
                transformed[2][dst] = (a - b + c) * 0.5;
                transformed[3][dst] = c;
            }
        }
        Ok(Self { channels_in, channels_out, transformed })
    }

    pub fn forward_same(&self, input: &[f32], length: usize, tile_pairs: usize) -> Vec<f32> {
        assert!(tile_pairs > 0);
        assert_eq!(input.len(), self.channels_in * length);
        let (ci, co) = (self.channels_in, self.channels_out);
        let mut output = vec![0f32; co * length];
        if length == 0 { return output; }
        let pairs = length.div_ceil(2);
        let tile_pairs = tile_pairs.min(pairs);
        let mut inputs: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0f32; ci * tile_pairs]);
        let mut products: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0f32; co * tile_pairs]);
        for base in (0..pairs).step_by(tile_pairs) {
            let n = (pairs - base).min(tile_pairs);
            for c in 0..ci {
                let row = &input[c * length..(c + 1) * length];
                for j in 0..n {
                    let center = 2 * (base + j);
                    let sample = |offset: usize| -> f32 {
                        let index = center + offset;
                        if index >= 1 && index - 1 < length { row[index - 1] } else { 0.0 }
                    };
                    let (a,b,c2,d) = (sample(0),sample(1),sample(2),sample(3));
                    let ix = c * n + j;
                    inputs[0][ix] = a - c2;
                    inputs[1][ix] = b + c2;
                    inputs[2][ix] = c2 - b;
                    inputs[3][ix] = b - d;
                }
            }
            for phase in 0..4 {
                unsafe {
                    gemm::gemm(co,n,ci,
                        products[phase].as_mut_ptr(),1,n as isize,false,
                        self.transformed[phase].as_ptr(),1,ci as isize,
                        inputs[phase].as_ptr(),1,n as isize,
                        0f32,1f32,false,false,false,gemm::Parallelism::None);
                }
            }
            for oc in 0..co {
                for j in 0..n {
                    let index = 2*(base+j);
                    let p = oc*n+j;
                    output[oc*length+index] = products[0][p]+products[1][p]+products[2][p];
                    if index + 1 < length {
                        output[oc*length+index+1] = products[1][p]-products[2][p]-products[3][p];
                    }
                }
            }
        }
        output
    }
}

fn native_conv1d_k3(input:&[f32],weights:&[f32],ci:usize,co:usize,length:usize)->Vec<f32>{
    let mut output=vec![0f32;co*length];
    for tap in 0..3 {
        let start=1usize.saturating_sub(tap).min(length);
        let end=(length+1).saturating_sub(tap).min(length);
        if start>=end {continue;}
        let first=start+tap-1;
        unsafe {
            gemm::gemm(co,end-start,ci,output.as_mut_ptr().add(start),1,length as isize,true,
                weights.as_ptr().add(tap),3,(ci*3) as isize,
                input.as_ptr().add(first),1,length as isize,
                1f32,1f32,false,false,false,gemm::Parallelism::None);
        }
    }
    output
}

fn median(mut data:Vec<f64>)->f64{ data.sort_by(f64::total_cmp);data[data.len()/2] }
fn raw(path:&str)->Vec<f32>{
    let bytes=std::fs::read(path).unwrap();
    assert_eq!(bytes.len()%4,0);
    bytes.chunks_exact(4).map(|chunk|f32::from_le_bytes(chunk.try_into().unwrap())).collect()
}
fn benchmark_case(dir:&str,ci:usize,co:usize,length:usize){
    let input=raw(&format!("{dir}/x.bin"));
    let weights=raw(&format!("{dir}/w.bin"));
    let expected=raw(&format!("{dir}/out.bin"));
    let bias=raw(&format!("{dir}/bias.bin"));
    assert_eq!(input.len(),ci*length);
    let plan=F23Plan::from_oik3(&weights,ci,co).unwrap();
    let original=native_conv1d_k3(&input,&weights,ci,co,length);
    for tile in [16,32,64,128,256] {
        let result=plan.forward_same(&input,length,tile);
        let mut max_abs=0f32;
        let mut max_torch=0f32;
        for (i,(&reference,&actual)) in original.iter().zip(&result).enumerate(){
            max_abs=max_abs.max((reference-actual).abs());
            max_torch=max_torch.max((actual+bias[i/length]-expected[i]).abs());
        }
        assert!(max_abs<0.0001,"numeric error {max_abs}");
        let mut old=Vec::new();let mut new=Vec::new();
        for _ in 0..3 {black_box(native_conv1d_k3(&input,&weights,ci,co,length));black_box(plan.forward_same(&input,length,tile));}
        for _ in 0..9{
            let t=Instant::now();black_box(native_conv1d_k3(&input,&weights,ci,co,length));old.push(t.elapsed().as_secs_f64()*1000.0);
            let t=Instant::now();black_box(plan.forward_same(&input,length,tile));new.push(t.elapsed().as_secs_f64()*1000.0);
        }
        let (old,new)=(median(old),median(new));
        println!("tile={tile} old_ms={old:.5} winograd_ms={new:.5} gain={:.3}x max_abs={max_abs:.8} max_vs_torch={max_torch:.8}",old/new);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_f23_k3_finite() {
        for ci in [1usize,3,16,64] {for co in [1usize,5,32] {for len in [0usize,1,2,3,5,16,31,128] {
            let input=(0..ci*len).map(|i|((i as f32)*0.013).sin()*0.5).collect::<Vec<_>>();
            let weights=(0..co*ci*3).map(|i|((i as f32)*0.023).cos()*0.06).collect::<Vec<_>>();
            let original=native_conv1d_k3(&input,&weights,ci,co,len);
            let plan=F23Plan::from_oik3(&weights,ci,co).unwrap();
            for tile in [1,2,7,32] {
                let output=plan.forward_same(&input,len,tile);
                assert_eq!(original.len(),output.len());
                for (&a,&b) in original.iter().zip(&output){
                    assert!((a-b).abs()<0.0002,"ci={ci},co={co},length={len},tile={tile}: {a} != {b}");
                }
            }
        }}}
    }
    #[test]
    fn non_finite_weights_are_rejected(){
        assert!(F23Plan::from_oik3(&[1.0,f32::INFINITY,3.0],1,1).is_err());
    }
}

fn main(){
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()==5 {
        benchmark_case(&args[1],args[2].parse().unwrap(),args[3].parse().unwrap(),args[4].parse().unwrap());
    } else { eprintln!("usage: nsf_winograd_f23 PATH_TO_CASE CH_IN CH_OUT LENGTH"); }
}
