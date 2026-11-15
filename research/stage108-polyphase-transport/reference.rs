#[cfg(test)]
mod tests {
    fn check(batch: usize, cin: usize, cout: usize, groups: usize, time: usize, k: usize, stride: usize, pad: usize) {
        let x: Vec<f32> = (0..batch*cin*time).map(|i| (i as f32 * 0.017).sin()).collect();
        let w: Vec<f32> = (0..cin*(cout/groups)*k).map(|i| (i as f32 * 0.013).cos()).collect();
        let coutg = cout/groups;
        let cing = cin/groups;
        let len = (time-1)*stride - 2*pad + k;
        let mut packed = vec![0f32; w.len()];
        for ci in 0..cin { for co in 0..coutg { for ki in 0..k {
            packed[(ki*cin+ci)*coutg+co] = w[(ci*coutg+co)*k+ki];
        }}}
        let mut ntc = vec![0f32; x.len()];
        for b in 0..batch { for ci in 0..cin { for t in 0..time {
            ntc[(b*time+t)*cin+ci] = x[(b*cin+ci)*time+t];
        }}}
        for b in 0..batch { for co in 0..cout { for t in 0..len {
            let group = co / coutg;
            let co_local = co % coutg;
            let base = t + pad;
            let phase = base % stride;
            let source_time = base / stride;
            let taps = (k+stride-1-phase)/stride;
            let mut direct = 0f32;
            let mut kic_strided = 0f32;
            let mut kic_ntc = 0f32;
            for ti in 0..taps {
                if source_time < ti || source_time - ti >= time {continue;}
                let source = source_time - ti;
                let ki = phase + ti*stride;
                for ci in group*cing..(group+1)*cing {
                    let w0 = w[(ci*coutg+co_local)*k+ki];
                    let w1 = packed[(ki*cin+ci)*coutg+co_local];
                    direct += x[(b*cin+ci)*time+source]*w0;
                    kic_strided += x[(b*cin+ci)*time+source]*w1;
                    kic_ntc += ntc[(b*time+source)*cin+ci]*w1;
                }
            }
            assert!((direct-kic_strided).abs() < 1e-5);
            assert!((direct-kic_ntc).abs() < 1e-5);
        }}}
    }
    #[test] fn dense_upsample8() {check(2,64,32,1,32,16,8,4);}
    #[test] fn dense_upsample10() {check(2,128,64,1,19,20,10,5);}
    #[test] fn odd_stride_with_tail() {check(1,7,12,1,13,9,4,2);}
    #[test] fn grouped() {check(1,12,16,4,21,7,3,2);}
    #[test] fn stride_one() {check(1,5,7,1,15,3,1,1);}
}
