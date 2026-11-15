#[derive(Clone, Copy)]
struct Shape { n: usize, h: usize, w: usize, cin: usize, cout: usize }

fn input_value(i: usize) -> f32 { ((i * 37 % 149) as f32 - 70.0) * 0.005 }
fn weight_value(i: usize) -> f32 { ((i * 53 % 211) as f32 - 90.0) * 0.001 }

fn compute(shape: Shape, pair: bool) -> Vec<f32> {
    let Shape { n, h, w, cin, cout } = shape;
    let input: Vec<f32> = (0..n*h*w*cin).map(input_value).collect();
    let weight: Vec<f32> = (0..cout*3*3*cin).map(weight_value).collect();
    let mut output = vec![0.0f32; n*h*w*cout];
    let at = |batch: usize, y: i32, x: i32, ci: usize| {
        if y < 0 || x < 0 || y >= h as i32 || x >= w as i32 { 0.0 }
        else { input[((batch*h + y as usize)*w + x as usize)*cin + ci] }
    };
    let wgt = |co:usize,kh:usize,kw:usize,ci:usize| { weight[((co*3 + kh)*3 + kw)*cin + ci] };
    if !pair {
        for batch in 0..n { for y in 0..h { for x in 0..w { for co in 0..cout {
            let mut sum = co as f32 * 0.002;
            for kh in 0..3 { for kw in 0..3 { for ci in 0..cin {
                sum += at(batch, y as i32 + kh as i32 - 1, x as i32 + kw as i32 - 1, ci) * wgt(co,kh,kw,ci);
            }}}
            output[((batch*h+y)*w+x)*cout+co] = sum;
        }}}}
    } else {
        for batch in 0..n { for y in 0..h { for x in (0..w).step_by(2) { for co in 0..cout {
            let mut accum = [co as f32 * 0.002; 2];
            for ci in 0..cin {
                let mut footprint = [0.0f32; 12];
                for kh in 0..3 { for col in 0..4 {
                    footprint[kh*4+col] = at(batch,y as i32 + kh as i32 - 1,x as i32 + col as i32 - 1,ci);
                }}
                for kh in 0..3 { for kw in 0..3 {
                    let weight = wgt(co,kh,kw,ci);
                    accum[0] += footprint[kh*4+kw] * weight;
                    accum[1] += footprint[kh*4+kw+1] * weight;
                }}
            }
            output[((batch*h+y)*w+x)*cout+co] = accum[0];
            if x+1 < w { output[((batch*h+y)*w+x+1)*cout+co] = accum[1]; }
        }}}}
    }
    output
}

#[test]
fn odd_even_and_boundary_exactness() {
    for shape in [
        Shape {n:1,h:1,w:1,cin:1,cout:1},
        Shape {n:1,h:1,w:5,cin:3,cout:4},
        Shape {n:1,h:3,w:1,cin:4,cout:8},
        Shape {n:1,h:3,w:5,cin:3,cout:5},
        Shape {n:2,h:6,w:6,cin:8,cout:8},
        Shape {n:1,h:7,w:9,cin:16,cout:16},
        Shape {n:1,h:11,w:8,cin:32,cout:8},
    ] {
        let direct = compute(shape,false);
        let pair = compute(shape,true);
        let max = direct.iter().zip(pair.iter()).map(|(x,y)| (x-y).abs()).fold(0.0f32,f32::max);
        println!("shape {:?}, max_abs={max}",(shape.n,shape.h,shape.w,shape.cin,shape.cout));
        assert!(max <= 2e-5, "max_abs={}", max);
    }
}

#[test]
fn tile_footprint_and_tradeoff() {
    let direct_reads_per_two_outputs = 2*3*3;
    let pair_reads_per_two_outputs = 3*4;
    let reuse_reads_per_four_outputs = 4*4;
    assert_eq!((direct_reads_per_two_outputs,pair_reads_per_two_outputs,reuse_reads_per_four_outputs),(18,12,16));
    assert!(pair_reads_per_two_outputs < direct_reads_per_two_outputs);
    assert!(pair_reads_per_two_outputs*2 > reuse_reads_per_four_outputs);
}
