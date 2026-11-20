#[cfg(test)]
mod tests {
    fn validate(
        batch: usize,
        cin: usize,
        cout: usize,
        groups: usize,
        time: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) {
        let out_group = cout / groups;
        let in_group = cin / groups;
        let output_length = (time - 1) * stride + kernel - 2 * padding;
        let x: Vec<f32> = (0..batch * cin * time)
            .map(|i| (i as f32 * 0.031).sin() * 0.2)
            .collect();
        let w: Vec<f32> = (0..cin * out_group * kernel)
            .map(|i| (i as f32 * 0.027).cos() * 0.1)
            .collect();
        for n in 0..batch {
            for oc in 0..cout {
                let group = oc / out_group;
                let local_oc = oc % out_group;
                for t in 0..output_length {
                    let base = t + padding;
                    let phase = base % stride;
                    let source_time = base / stride;
                    let taps = (kernel + stride - phase - 1) / stride;
                    let mut reference = (oc as f32) * 0.001;
                    let mut ordered = reference;
                    let mut parts = [0f32; 4];
                    for tap in 0..taps {
                        if source_time < tap || source_time - tap >= time {
                            continue;
                        }
                        let src = source_time - tap;
                        let ki = phase + tap * stride;
                        for ci in 0..in_group {
                            let channel = group * in_group + ci;
                            let xv = x[(n * cin + channel) * time + src];
                            let weight = w[(channel * out_group + local_oc) * kernel + ki];
                            let product = xv * weight;
                            reference += product;
                            ordered += product;
                            parts[ci % 4] += product;
                        }
                    }
                    let rearranged =
                        (parts[0] + parts[1]) + (parts[2] + parts[3]) + (oc as f32) * 0.001;
                    assert_eq!(reference.to_bits(), ordered.to_bits());
                    assert!(
                        (reference - rearranged).abs() < 1e-4,
                        "mismatch n={n} oc={oc} t={t} direct={reference} ilp4={rearranged}"
                    );
                }
            }
        }
    }

    #[test]
    fn upsample8() {
        validate(2, 64, 32, 1, 25, 16, 8, 4)
    }
    #[test]
    fn upsample10() {
        validate(2, 128, 64, 1, 19, 20, 10, 5)
    }
    #[test]
    fn grouped() {
        validate(1, 12, 16, 4, 23, 7, 3, 2)
    }
    #[test]
    fn nonmultiple_of_four() {
        validate(1, 7, 12, 1, 13, 9, 4, 3)
    }
    #[test]
    fn one_channel() {
        validate(1, 1, 5, 1, 22, 3, 2, 1)
    }
    #[test]
    fn stride_one() {
        validate(1, 11, 9, 1, 12, 3, 1, 1)
    }
}
