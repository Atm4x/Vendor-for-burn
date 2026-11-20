fn activate(x: f32) -> f32 {
    if x >= 0.0 { x } else { x * 0.1 }
}

fn direct_ref(
    input: &[f32], weight: &[f32], bias: &[f32], residual: &[f32],
    time: usize, cin: usize, cout: usize, kernel: usize, dilation: usize,
    preactivated: bool,
) -> Vec<f32> {
    let pad = ((kernel - 1) * dilation / 2) as isize;
    let mut out = vec![0f32; time * cout];
    for t in 0..time {
        for co in 0..cout {
            let mut accum = 0.0f32;
            for k in 0..kernel {
                let ti = t as isize + (k * dilation) as isize - pad;
                if !(0..time as isize).contains(&ti) { continue; }
                for ci in 0..cin {
                    let x = input[ti as usize * cin + ci];
                    let x = if preactivated { x } else { activate(x) };
                    accum = weight[(k * cin + ci) * cout + co].mul_add(x, accum);
                }
            }
            out[t * cout + co] = (accum + bias[co]) + residual[t * cout + co];
        }
    }
    out
}

fn tile4_fast_interior_linear(
    input: &[f32], weight: &[f32], bias: &[f32], residual: &[f32],
    time: usize, cin: usize, cout: usize, kernel: usize, dilation: usize,
    preactivated: bool,
) -> Vec<f32> {
    let pad = ((kernel - 1) * dilation / 2) as isize;
    let mut out = vec![0f32; time * cout];
    for t0 in (0..time).step_by(4) {
        let first = t0 as isize - pad;
        let last = (t0 + 3) as isize + ((kernel - 1) * dilation) as isize - pad;
        let interior = first >= 0 && last < time as isize && t0 + 3 < time;
        for co in 0..cout {
            let mut sum = [0f32; 4];
            for k in 0..kernel {
                let i0 = t0 as isize + (k * dilation) as isize - pad;
                let mut w_offset = (k * cin) * cout + co;
                if interior {
                    let mut input_index = [0usize; 4];
                    for dt in 0..4 {
                        input_index[dt] = (i0 as usize + dt) * cin;
                    }
                    for _ in 0..cin {
                        let w = weight[w_offset];
                        for dt in 0..4 {
                            let x = input[input_index[dt]];
                            let x = if preactivated { x } else { activate(x) };
                            sum[dt] = w.mul_add(x, sum[dt]);
                            input_index[dt] += 1;
                        }
                        w_offset += cout;
                    }
                } else {
                    for ci in 0..cin {
                        let w = weight[w_offset];
                        for dt in 0..4 {
                            let t = t0 + dt;
                            let ti = i0 + dt as isize;
                            if t < time && (0..time as isize).contains(&ti) {
                                let x = input[ti as usize * cin + ci];
                                let x = if preactivated { x } else { activate(x) };
                                sum[dt] = w.mul_add(x, sum[dt]);
                            }
                        }
                        w_offset += cout;
                    }
                }
            }
            for dt in 0..4 {
                let t = t0 + dt;
                if t < time {
                    out[t * cout + co] = (sum[dt] + bias[co]) + residual[t * cout + co];
                }
            }
        }
    }
    out
}

fn interior_tiles(time: usize, kernel: usize, dilation: usize) -> (usize,usize) {
    let pad=((kernel-1)*dilation/2) as isize;
    let mut ok=0;let mut total=0;
    for t0 in (0..time).step_by(4) {
        total+=1;
        if t0 as isize -pad>=0 && (t0+3+(kernel-1)*dilation) as isize-pad < time as isize &&t0+3<time {ok+=1;}
    }
    (ok,total)
}

#[test]
fn test_linear_interior_bitwise_matches_direct() {
    for &time in &[1,2,3,4,5,17,33,71] {
        for &(cin,cout) in &[(1,1),(3,4),(16,32),(128,128)] {
            for &(kernel,dilation) in &[(3,1),(7,3),(11,1),(11,5)] {
                for &preactivated in &[false,true] {
                    let input:Vec<f32>=(0..time*cin).map(|i| ((i*19%91) as f32-45.)*0.013).collect();
                    let weight:Vec<f32>=(0..kernel*cin*cout).map(|i|((i*11%89) as f32-44.)*0.006).collect();
                    let bias:Vec<f32>=(0..cout).map(|i|((i*7%61) as f32-30.)*0.001).collect();
                    let residual:Vec<f32>=(0..time*cout).map(|i|((i*17%53) as f32-26.)*0.002).collect();
                    let expected=direct_ref(&input,&weight,&bias,&residual,time,cin,cout,kernel,dilation,preactivated);
                    let actual=tile4_fast_interior_linear(&input,&weight,&bias,&residual,time,cin,cout,kernel,dilation,preactivated);
                    for (i,(a,b)) in expected.iter().zip(actual.iter()).enumerate() {
                        assert_eq!(a.to_bits(),b.to_bits(),"T={time} cin={cin} cout={cout} K={kernel} D={dilation} preact={preactivated} pos={i}");
                    }
                }
            }
        }
    }
}

#[test]
fn test_interior_coverage_on_actual_resblock_length() {
    for &dilation in &[1,3,5] {
        let (ok,total)=interior_tiles(29_800,11,dilation);
        assert_eq!(total,7_450);
        assert!(ok as f64/total as f64>0.998);
        println!("C128/K11 D={dilation} interior tiles={ok}/{total} ({:.4}%)",100.*ok as f64/total as f64);
    }
}
