use crate::resample;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

const SAMPLE_RATE: u32 = 16000;
const FFT_SIZE: usize = 8192;
const HOP: usize = 1024;
const MIN_CENTS: f64 = 10.0;
const MIN_STRENGTH: f64 = 0.12;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tuning {
    pub cents: f64,
    pub strength: f64,
    /// Correction applied to the input's pitch (zero when disabled).
    pub applied_cents: f64,
    /// Duration of the original signal / duration of the retuned signal.
    pub time_scale: f64,
}

impl Tuning {
    pub fn identity() -> Self {
        Self { cents: 0.0, strength: 0.0, applied_cents: 0.0, time_scale: 1.0 }
    }
}

/// The upstream 5-cent harmonic-salience estimator (six weighted harmonics).
/// Center-free STFT with a symmetric Hann window on 8192-sample frames.
pub fn estimate(audio16k: &[f32]) -> Result<Tuning, String> {
    if audio16k.iter().any(|x| !x.is_finite()) { return Err("non-finite audio".into()); }
    if audio16k.len() < FFT_SIZE { return Ok(Tuning::identity()); }
    let nframes=1+(audio16k.len()-FFT_SIZE)/HOP;
    let first=1200.0*(65.0f64/440.0).log2();
    let last=1200.0*(1000.0f64/440.0).log2();
    let mut cents=Vec::new();
    let mut grid=first;
    while grid < last { cents.push(grid); grid+=5.0; }
    let fundamentals=cents.iter().map(|&c| 440.0*2f64.powf(c/1200.0))
        .collect::<Vec<_>>();
    let win=(0..FFT_SIZE).map(|i| (0.5-0.5*(2.0*PI*i as f64/(FFT_SIZE-1) as f64).cos()) as f32)
        .collect::<Vec<_>>();
    let mut planner=FftPlanner::<f32>::new();
    let fft=planner.plan_fft_forward(FFT_SIZE);
    let mut buffer=vec![Complex::new(0.0f32,0.0f32);FFT_SIZE];
    let mut magnitude=vec![0.0f32;FFT_SIZE/2+1];
    let mut salience=vec![0.0f32;fundamentals.len()];
    let mut sum_real=0.0f64;
    let mut sum_imag=0.0f64;
    let mut total_weight=0.0f64;
    for frame in 0..nframes {
        let start=frame*HOP;
        for i in 0..FFT_SIZE {
            buffer[i].re=audio16k[start+i]*win[i];
            buffer[i].im=0.0;
        }
        fft.process(&mut buffer);
        for (i,m) in magnitude.iter_mut().enumerate(){*m=buffer[i].norm().sqrt();}
        salience.fill(0.0);
        for harmonic in 1..=6 {
            let scale=0.85f32.powi(harmonic-1);
            for (i,&fundamental) in fundamentals.iter().enumerate() {
                let bin=harmonic as f64 * fundamental * FFT_SIZE as f64 / SAMPLE_RATE as f64;
                if bin >= magnitude.len() as f64 - 2.0 {continue;}
                let lower=bin.floor() as usize;
                let frac=(bin-lower as f64) as f32;
                salience[i]+=scale*(magnitude[lower]*(1.0-frac)+magnitude[lower+1]*frac);
            }
        }
        let max=salience.iter().copied().fold(0.0f32,f32::max);
        if max<=0.0 {continue;}
        for i in 1..salience.len()-1 {
            let (a,b,c)=(salience[i-1],salience[i],salience[i+1]);
            if !(b>a && b>=c && b>0.5*max) {continue;}
            let denom=a as f64 - 2.0*b as f64 + c as f64 - 1e-12;
            let adjustment=(0.5*(a as f64-c as f64)/denom).clamp(-1.0,1.0);
            let cent=(cents[i]+adjustment*5.0+50.0).rem_euclid(100.0)-50.0;
            let angle=2.0*PI*cent/100.0;
            sum_real+=b as f64*angle.cos();
            sum_imag+=b as f64*angle.sin();
            total_weight+=b as f64;
        }
    }
    if total_weight<=0.0 {return Ok(Tuning::identity());}
    let cent=sum_imag.atan2(sum_real)*100.0/(2.0*PI);
    let strength=sum_real.hypot(sum_imag)/total_weight;
    let shift=if cent.abs()>=MIN_CENTS && strength>=MIN_STRENGTH{-cent}else{0.0};
    Ok(Tuning{cents:cent,strength,applied_cents:shift,
        time_scale:2.0f64.powf(shift/1200.0)})
}

pub fn normalize(audio16k: &[f32]) -> Result<(Vec<f32>, Tuning), String> {
    let info=estimate(audio16k)?;
    let signal=if info.applied_cents!=0.0 {
        resample::resample(audio16k,SAMPLE_RATE,SAMPLE_RATE as f64/info.time_scale)?
    } else {audio16k.to_vec()};
    Ok((signal,info))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn silence_has_no_tuning_correction(){
        let zero=vec![0.0; 16384];
        let (retuned,info)=normalize(&zero).unwrap();
        assert_eq!(info.time_scale,1.0);
        assert_eq!(retuned,zero);
    }
    #[test]
    fn exact_a440_needs_no_large_correction(){
        let sine=(0..32000).map(|i|
            (0.25*(2.0*PI*440.0*i as f64/16000.0).sin()) as f32).collect::<Vec<_>>();
        let tune=estimate(&sine).unwrap();
        assert!(tune.cents.abs()<10.0,"estimated {} cents",tune.cents);
        assert_eq!(tune.applied_cents,0.0);
    }
    #[test]
    fn harmonic_tuner_detects_twenty_seven_cent_shift() {
        let rate=16000.0;
        let f=447.0;
        let audio=(0..32000).map(|i| {
            let t=i as f64/rate;
            (0.24*(2.0*PI*f*t).sin()+0.08*(2.0*PI*2.0*f*t).sin()) as f32
        }).collect::<Vec<_>>();
        let info=estimate(&audio).unwrap();
        assert!((info.cents-27.040613).abs()<1.0,
            "expected +27 cents, got {}",info.cents);
        assert!(info.strength>0.9);
        assert!((info.applied_cents+info.cents).abs()<1e-8);
    }

}
