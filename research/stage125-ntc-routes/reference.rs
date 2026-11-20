fn routed_times(t: usize, d: usize, tile: usize, residue: bool) -> Vec<usize> {
    let mut times = Vec::new();
    let tiles = if residue { d * (t.div_ceil(d)).div_ceil(tile) } else { t.div_ceil(tile) };
    for work in 0..tiles {
        let (start, step) = if residue {
            let per_residue = t.div_ceil(d).div_ceil(tile);
            let class = work / per_residue;
            let local = work % per_residue;
            ((local * tile) * d + class, d)
        } else { (work * tile, 1) };
        for lane in 0..tile {
            let pos = start + lane * step;
            if pos < t { times.push(pos); }
        }
    }
    times
}

fn act(v: f32) -> f32 { if v >= 0.0 { v } else { v * 0.1 } }

#[allow(clippy::too_many_arguments)]
fn conv(
    input: &[f32], weights: &[f32], bias: &[f32], shortcut: Option<&[f32]>,
    time: usize, cin: usize, cout: usize, kernel: usize, d: usize,
    activated_input: bool, activated_output: bool, residue: bool,
) -> Vec<f32> {
    let mut result = vec![0.0f32; time * cout];
    let pad = ((kernel - 1) * d / 2) as isize;
    for t in routed_times(time, d, 4, residue) {
        for co in 0..cout {
            let mut sum = 0.0f32;
            for k in 0..kernel {
                let index = t as isize + (k * d) as isize - pad;
                if index < 0 || index >= time as isize { continue; }
                for ci in 0..cin {
                    let input_value = input[index as usize * cin + ci];
                    let a = if activated_input { input_value } else { act(input_value) };
                    let weight = weights[(k * cin + ci) * cout + co];
                    sum = weight.mul_add(a, sum);
                }
            }
            sum += bias[co];
            if let Some(shortcut) = shortcut { sum += shortcut[t * cout + co]; }
            if activated_output { sum = act(sum); }
            result[t * cout + co] = sum;
        }
    }
    result
}

#[test]
fn residue_schedule_is_a_bijection_including_short_tails() {
    for t in [1,2,3,4,5,6,7,13,17,30,71,129,29_800] {
        for d in [1,3,5,9] {
            let mut got = routed_times(t,d,4,true);
            got.sort_unstable();
            assert_eq!(got, (0..t).collect::<Vec<_>>(),"T={t} D={d}");
        }
    }
}

#[test]
fn residue_first_conv_and_second_conv_preserve_every_f32_bit() {
    for time in [1usize,2,3,4,5,17,71] {
        for channels in [1usize,3,8,16,128] {
            for kernel in [3usize,7,11] {
                for dilation in [3usize,5] {
                    let input = (0..time*channels).map(|i| ((i*29 % 107) as f32 - 51.0)*0.004).collect::<Vec<_>>();
                    let weights = (0..kernel*channels*channels)
                        .map(|i| ((i*17%59) as f32-29.0)*0.002).collect::<Vec<_>>();
                    let weights2 = (0..kernel*channels*channels)
                        .map(|i| ((i*13%67) as f32-33.0)*0.001).collect::<Vec<_>>();
                    let bias = (0..channels).map(|i| i as f32*0.0002-0.006).collect::<Vec<_>>();
                    let direct = conv(&input,&weights,&bias,None,time,channels,channels,kernel,dilation,false,true,false);
                    let routed = conv(&input,&weights,&bias,None,time,channels,channels,kernel,dilation,false,true,true);
                    for (a,b) in direct.iter().zip(&routed) { assert_eq!(a.to_bits(),b.to_bits()); }
                    let direct2 = conv(&direct,&weights2,&bias,Some(&input),time,channels,channels,kernel,1,true,false,false);
                    let routed2 = conv(&routed,&weights2,&bias,Some(&input),time,channels,channels,kernel,1,true,false,false);
                    for (a,b) in direct2.iter().zip(&routed2) { assert_eq!(a.to_bits(),b.to_bits()); }
                }
            }
        }
    }
}

#[test]
fn route_overhead_is_bounded() {
    for t in [1usize,3,17,71,29800] {
        for d in [3usize,5] {
            let ref_tiles=t.div_ceil(4);
            let residue_tiles=d*t.div_ceil(d).div_ceil(4);
            assert!(residue_tiles>=ref_tiles);
            assert!(residue_tiles<=ref_tiles+d);
        }
    }
}
