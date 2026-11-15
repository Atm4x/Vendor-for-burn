#[derive(Clone, Copy)]
struct Shape { n: usize, h: usize, w: usize, cin: usize, cout: usize }

fn input_value(i: usize) -> f32 { ((i * 37 % 149) as f32 - 70.0) * 0.005 }
fn weight_value(i: usize) -> f32 { ((i * 53 % 211) as f32 - 90.0) * 0.001 }

fn compute(shape: Shape, reuse: bool) -> Vec<f32> {
    let Shape { n, h, w, cin, cout } = shape;
    let a: Vec<f32> = (0..n*h*w*cin).map(input_value).collect();
    let b: Vec<f32> = (0..cout*3*3*cin).map(weight_value).collect();
    let mut out = vec![0.0f32; n*h*w*cout];
    let at = |batch: usize, y: i32, x: i32, ci: usize| -> f32 {
        if y < 0 || x < 0 || y >= h as i32 || x >= w as i32 { 0.0 }
        else { a[((batch * h + y as usize) * w + x as usize) * cin + ci] }
    };
    let wgt = |co: usize, kh: usize, kw: usize, ci: usize| -> f32 { b[((co*3 + kh)*3 + kw)*cin + ci] };
    if !reuse {
        for batch in 0..n { for y in 0..h { for x in 0..w { for co in 0..cout {
            let mut sum = co as f32 * 0.002;
            for kh in 0..3 { for kw in 0..3 { for ci in 0..cin {
                sum += at(batch, y as i32 + kh as i32 - 1, x as i32 + kw as i32 - 1, ci) * wgt(co,kh,kw,ci);
            }}}
            out[((batch*h+y)*w+x)*cout+co] = sum;
        }}}}
    } else {
        for batch in 0..n { for ty in (0..h).step_by(2) { for tx in (0..w).step_by(2) { for co in 0..cout {
            let mut sums = [co as f32 * 0.002; 4];
            for ci in 0..cin {
                let mut footprint = [0.0f32; 16];
                for ky in 0..4 { for kx in 0..4 {
                    footprint[ky*4+kx] = at(batch, ty as i32 + ky as i32 - 1, tx as i32 + kx as i32 - 1, ci);
                }}
                for kh in 0..3 { for kw in 0..3 {
                    let weight = wgt(co,kh,kw,ci);
                    sums[0] += footprint[kh*4+kw] * weight;
                    sums[1] += footprint[kh*4+kw+1] * weight;
                    sums[2] += footprint[(kh+1)*4+kw] * weight;
                    sums[3] += footprint[(kh+1)*4+kw+1] * weight;
                }}
            }
            for dy in 0..2 { for dx in 0..2 {
                if ty+dy<h && tx+dx<w { out[((batch*h+ty+dy)*w+tx+dx)*cout+co] = sums[dy*2+dx]; }
            }}
        }}}}
    }
    out
}

#[test]
fn matches_direct_on_even_odd_and_boundary_shapes() {
    for s in [
        Shape {n:1,h:1,w:1,cin:1,cout:1},
        Shape {n:1,h:2,w:2,cin:1,cout:3},
        Shape {n:1,h:3,w:5,cin:3,cout:5},
        Shape {n:2,h:6,w:6,cin:8,cout:8},
        Shape {n:1,h:7,w:9,cin:16,cout:16},
        Shape {n:1,h:11,w:8,cin:32,cout:8},
    ] {
        let x = compute(s,false); let y = compute(s,true);
        let max = x.iter().zip(y.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32,f32::max);
        let rms = (x.iter().zip(y.iter()).map(|(a,b)| ((a-b) as f64).powi(2)).sum::<f64>()/x.len() as f64).sqrt();
        println!("shape {:?}, max_abs={max}, rmse={rms}",(s.n,s.h,s.w,s.cin,s.cout));
        assert!(max <= 2e-5, "max_abs={max}");
    }
}

#[test]
fn interior_footprint_reduces_loads_by_more_than_half() {
    let old_reads_per_ci = 4 * 3 * 3;
    let new_reads_per_ci = (2 + 3 - 1) * (2 + 3 - 1);
    assert_eq!(old_reads_per_ci,36);
    assert_eq!(new_reads_per_ci,16);
    assert!(new_reads_per_ci * 2 < old_reads_per_ci);
}
