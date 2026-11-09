use std::{hint::black_box,time::Instant};
struct Plan {ci:usize,co:usize,k:usize,segments:usize, g:[Vec<f32>;4]}
impl Plan{
 fn new(w:&[f32],ci:usize,co:usize,k:usize)->Self{
  assert!(k%2==1&&k>0&&w.len()==ci*co*k);
  let segments=k.div_ceil(3);let reduction=segments*ci;
  let mut g: [Vec<f32>;4]=std::array::from_fn(|_|vec![0.0;co*reduction]);
  for oc in 0..co{for seg in 0..segments{for c in 0..ci{
   let at=|off:usize|->f32{ if seg*3+off<k {w[(oc*ci+c)*k+seg*3+off]} else {0.0}};
   let (a,b,v)=(at(0),at(1),at(2));let j=oc*reduction+seg*ci+c;
   g[0][j]=a;g[1][j]=(a+b+v)*0.5;g[2][j]=(a-b+v)*0.5;g[3][j]=v;
  }}}
  Self{ci,co,k,segments,g}
 }
 fn run(&self,x:&[f32],len:usize,dilation:usize,tile:usize)->Vec<f32>{
  assert!(dilation>0 && tile>0 && x.len()==self.ci*len);
  let mut y=vec![0f32;self.co*len];if len==0{return y;}
  let pad=(self.k-1)*dilation/2;
  let mut pairs=Vec::with_capacity(len.div_ceil(2)+dilation);
  for r in 0..dilation.min(len){let span=1+(len-1-r)/dilation;
   for q in (0..span).step_by(2) {let o0=r+q*dilation;let o1=if q+1<span{Some(o0+dilation)}else{None};pairs.push((o0,o1));}
  }
  let rows=self.segments*self.ci;
  let mut panels:[Vec<f32>;4]=std::array::from_fn(|_|vec![0.0;rows*tile]);
  let mut products:[Vec<f32>;4]=std::array::from_fn(|_|vec![0.0;self.co*tile]);
  for base in (0..pairs.len()).step_by(tile){
   let n=(pairs.len()-base).min(tile);
   for s in 0..self.segments{for c in 0..self.ci{
    let ix=s*self.ci+c;let src=&x[c*len..(c+1)*len];
    for j in 0..n{
     let (o0,_)=pairs[base+j];
     let at=|tap:usize|->f32 {let p=o0 as isize+(3*s+tap) as isize*dilation as isize-pad as isize;
      if p>=0&&(p as usize)<len {src[p as usize]}else{0.0}};
     let (a,b,v,dd)=(at(0),at(1),at(2),at(3));let p=ix*n+j;
     panels[0][p]=a-v;panels[1][p]=b+v;panels[2][p]=v-b;panels[3][p]=b-dd;
    }
   }}
   for phase in 0..4 {unsafe {gemm::gemm(self.co,n,rows,products[phase].as_mut_ptr(),1,n as isize,false,self.g[phase].as_ptr(),1,rows as isize,panels[phase].as_ptr(),1,n as isize,0f32,1f32,false,false,false,gemm::Parallelism::None);}}
   for oc in 0..self.co{for j in 0..n{
    let p=oc*n+j;let (o0,o1)=pairs[base+j];
    y[oc*len+o0]=products[0][p]+products[1][p]+products[2][p];
    if let Some(o1)=o1{y[oc*len+o1]=products[1][p]-products[2][p]-products[3][p];}
   }}
  }
  y
 }
}
fn direct(x:&[f32],w:&[f32],ci:usize,co:usize,lin:usize,k:usize,d:usize)->Vec<f32>{
 let p=(k-1)*d/2;let mut out=vec![0f32;co*lin];
 for ki in 0..k{let a=p.saturating_sub(ki*d).min(lin);let z=(lin+p).saturating_sub(ki*d).min(lin);if a>=z{continue;}let i0=a+ki*d-p;
 unsafe{gemm::gemm(co,z-a,ci,out.as_mut_ptr().add(a),1,lin as isize,true,w.as_ptr().add(ki),k as isize,(ci*k) as isize,x.as_ptr().add(i0),1,lin as isize,1f32,1f32,false,false,false,gemm::Parallelism::None);}
 }out
}
fn raw(p:&str)->Vec<f32>{let a=std::fs::read(p).unwrap();a.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect()}
fn med(mut v:Vec<f64>)->f64{v.sort_by(f64::total_cmp);v[v.len()/2]}
fn test(name:&str,ci:usize,co:usize,n:usize,k:usize,d:usize,x:&[f32],w:&[f32],truth:Option<&[f32]>,bias:Option<&[f32]>) {
 let reference=direct(x,w,ci,co,n,k,d);let plan=Plan::new(w,ci,co,k);
 println!("\n{name} ci={ci} n={n} k={k} d={d} segments={} theoretical_mul_ratio={:.3}",plan.segments, (2*plan.segments) as f64/k as f64);
 let mut ref_times=Vec::new();for _ in 0..3{black_box(direct(x,w,ci,co,n,k,d));}for _ in 0..7{let t=Instant::now();black_box(direct(x,w,ci,co,n,k,d));ref_times.push(t.elapsed().as_secs_f64()*1000.0);}let old=med(ref_times);
 println!("baseline_ms={old:.4}");
 for tile in [16,32,64,128,256,384]{
  let a=plan.run(x,n,d,tile);let err=reference.iter().zip(&a).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
  if let (Some(truth),Some(bias))=(truth,bias){let mx=a.iter().enumerate().map(|(i,b)|(b+bias[i/n]-truth[i]).abs()).fold(0f32,f32::max);print!("max_vs_pytorch={mx:.7} ");}
  if err>0.0005 {println!("tile={tile} BAD max_abs={err:.8}");continue;}
  let mut timings=Vec::new();for _ in 0..2{black_box(plan.run(x,n,d,tile));}for _ in 0..5{let t=Instant::now();black_box(plan.run(x,n,d,tile));timings.push(t.elapsed().as_secs_f64()*1000.0);}let new=med(timings);
  println!("tile={tile:3} ms={new:.4} speedup={:.3}x err={err:.8}",old/new);
 }
}
#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn finite_parity_many_shapes(){
  for ci in [1usize,3,16]{for co in [1usize,4,16]{for len in [1usize,2,3,11,25,83]{
   for k in [3usize,5,7,11]{for dilation in [1usize,2,5]{
    let x=(0..ci*len).map(|i|(i as f32*0.031).sin()*0.1).collect::<Vec<f32>>();
    let w=(0..co*ci*k).map(|i|(i as f32*0.017).cos()*0.008).collect::<Vec<f32>>();
    let reference=direct(&x,&w,ci,co,len,k,dilation);
    let plan=Plan::new(&w,ci,co,k);
    for tile in [1,3,16]{
     let actual=plan.run(&x,len,dilation,tile);
     for (a,b) in reference.iter().zip(actual.iter()){
      assert!((a-b).abs()<0.0002,"ci={ci} co={co} len={len} k={k} d={dilation} tile={tile} {a}!={b}");
     }
    }
   }}
  }}}
 }
}
fn main(){
 let args=std::env::args().collect::<Vec<_>>();
 if args.len()!=7 {
  eprintln!("usage: nsf_segmented_winograd PATH_TO_CASE C_IN C_OUT LEN K DILATION");
  return;
 }
 let (root,ci,co,n,k,d)=(&args[1],args[2].parse().unwrap(),args[3].parse().unwrap(),args[4].parse().unwrap(),args[5].parse().unwrap(),args[6].parse().unwrap());
 let x=raw(&format!("{root}/x.bin"));let w=raw(&format!("{root}/w.bin"));
 let expected=raw(&format!("{root}/out.bin"));let bias=raw(&format!("{root}/bias.bin"));
 test(root,ci,co,n,k,d,&x,&w,Some(&expected),Some(&bias));
}
