use audioadapter::Adapter;
use audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft, FixedSync, Resampler, WindowFunction};
use rustfft::{FftPlanner, num_complex::Complex32};
use std::{
    ffi::{CStr, c_char, c_uint, c_ulong, c_void},
    fs::File,
    io::Cursor,
    path::Path,
    sync::Arc,
};
use symphonia::core::{
    codecs::audio::AudioDecoderOptions,
    errors::Error,
    formats::{FormatOptions, TrackType, probe::Hint},
    io::{MediaSource, MediaSourceStream},
    meta::MetadataOptions,
};

const SAMPLE_RATE: u32 = 22_050;
const FFT_SIZE: usize = 1024;
const HOP: usize = 441;
const MEL_BINS: usize = 128;

pub struct MonoAudio {
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
}

pub fn decode(path: impl AsRef<Path>) -> Result<MonoAudio, String> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| e.to_string())?;
    decode_source(
        Box::new(file),
        path.extension().and_then(|x| x.to_str()),
        None,
    )
}

pub fn decode_snapshot(bytes: Vec<u8>, extension: &str) -> Result<MonoAudio, String> {
    decode_source(Box::new(Cursor::new(bytes)), Some(extension), Some(15 * 60))
}

fn decode_source(
    source: Box<dyn MediaSource>,
    extension: Option<&str>,
    duration_limit_secs: Option<usize>,
) -> Result<MonoAudio, String> {
    let source = MediaSourceStream::new(source, Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = extension {
        hint.with_extension(extension);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            source,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| format!("probing audio: {e}"))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or("no audio track")?;
    let id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .ok_or("missing codec parameters")?
        .audio()
        .ok_or("not audio")?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|e| format!("creating audio decoder: {e}"))?;
    let mut samples = Vec::new();
    let mut sample_rate = None;
    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(Error::ResetRequired) => return Err("audio stream format changed".into()),
            Err(e) => return Err(format!("reading audio packet: {e}")),
        };
        if packet.track_id != id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("decoding audio: {e}")),
        };
        let spec = decoded.spec();
        let channels = spec.channels().count();
        if channels == 0 {
            return Err("audio contains no channels".into());
        }
        if let Some(limit) = duration_limit_secs
            && (!(8_000..=192_000).contains(&spec.rate())
                || channels > 8
                || samples.len() + decoded.samples_interleaved() / channels
                    > limit * spec.rate() as usize)
        {
            return Err("audio exceeds supported stream limits".into());
        }
        if let Some(current) = sample_rate {
            if current != spec.rate() {
                return Err("sample rate changed midstream".into());
            }
        }
        sample_rate = Some(spec.rate());
        let mut interleaved = vec![0f32; decoded.samples_interleaved()];
        decoded.copy_to_slice_interleaved(&mut interleaved);
        samples.extend(
            interleaved
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32),
        );
    }
    let sample_rate = sample_rate.ok_or("empty audio stream")?;
    if samples.is_empty() {
        return Err("no audio samples".into());
    }
    Ok(MonoAudio {
        pcm: samples,
        sample_rate,
    })
}

#[repr(C)]
struct SoxrQuality {
    precision: f64,
    phase_response: f64,
    passband_end: f64,
    stopband_begin: f64,
    flags: c_ulong,
}

type SoxrSpec = unsafe extern "C" fn(c_ulong, c_ulong) -> SoxrQuality;
type SoxrOneshot = unsafe extern "C" fn(
    f64,
    f64,
    c_uint,
    *const c_void,
    usize,
    *mut usize,
    *mut c_void,
    usize,
    *mut usize,
    *const c_void,
    *const SoxrQuality,
    *const c_void,
) -> *const c_char;

fn soxr_resample(input: &[f32], source_rate: u32) -> Result<Vec<f32>, String> {
    #[cfg(target_os = "windows")]
    let libraries = ["soxr.dll", "libsoxr.dll"];
    #[cfg(target_os = "macos")]
    let libraries = ["libsoxr.dylib", "soxr.dylib"];
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    let libraries = ["libsoxr.so.0", "libsoxr.so"];

    let library = libraries
        .iter()
        .find_map(|&name| unsafe { libloading::Library::new(name).ok() })
        .ok_or("libsoxr is not installed")?;
    resample_with_soxr(input, source_rate, library)
}

pub fn resample_with_packaged_soxr(
    input: &[f32],
    source_rate: u32,
    library_path: &Path,
) -> Result<Vec<f32>, String> {
    if source_rate == SAMPLE_RATE {
        return Ok(input.to_vec());
    }
    let library = unsafe { libloading::Library::new(library_path) }
        .map_err(|error| format!("loading packaged SoXR: {error}"))?;
    resample_with_soxr(input, source_rate, library)
}

fn resample_with_soxr(
    input: &[f32],
    source_rate: u32,
    library: libloading::Library,
) -> Result<Vec<f32>, String> {
    let quality_spec: libloading::Symbol<'_, SoxrSpec> =
        unsafe { library.get(b"soxr_quality_spec") }.map_err(|e| e.to_string())?;
    let oneshot: libloading::Symbol<'_, SoxrOneshot> =
        unsafe { library.get(b"soxr_oneshot") }.map_err(|e| e.to_string())?;
    let quality = unsafe { quality_spec(4, 0) };
    let expected_len =
        ((input.len() as f64 * SAMPLE_RATE as f64 / source_rate as f64).round()) as usize;
    let mut output = vec![0.0f32; expected_len + 512];
    let mut input_consumed = 0usize;
    let mut output_written = 0usize;
    let error = unsafe {
        oneshot(
            source_rate as f64,
            SAMPLE_RATE as f64,
            1,
            input.as_ptr().cast::<c_void>(),
            input.len(),
            &mut input_consumed,
            output.as_mut_ptr().cast::<c_void>(),
            output.len(),
            &mut output_written,
            std::ptr::null(),
            &quality,
            std::ptr::null(),
        )
    };
    if !error.is_null() {
        return Err(unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned());
    }
    if input_consumed != input.len() || output_written < expected_len {
        return Err(format!(
            "soxr processed {input_consumed}/{} input samples and produced {output_written}/{expected_len} output samples",
            input.len()
        ));
    }
    output.truncate(expected_len);
    Ok(output)
}

fn rubato_resample(input: &[f32], source_rate: u32) -> Result<Vec<f32>, String> {
    let mut resampler = Fft::<f32>::new_custom(
        source_rate as usize,
        SAMPLE_RATE as usize,
        8192,
        1,
        1,
        WindowFunction::BlackmanHarris2,
        FixedSync::Input,
    )
    .map_err(|e| format!("resampler creation: {e}"))?;
    let planar = vec![input.to_vec()];
    let adapter = SequentialSliceOfVecs::new(&planar, 1, input.len()).map_err(|e| e.to_string())?;
    let result = resampler
        .process_all(&adapter, input.len(), None)
        .map_err(|e| e.to_string())?;
    Ok((0..result.frames())
        .map(|n| result.read_sample(0, n).unwrap_or(0.0))
        .collect())
}

pub fn resample_mono(input: &[f32], source_rate: u32) -> Result<Vec<f32>, String> {
    if source_rate == SAMPLE_RATE {
        return Ok(input.to_vec());
    }
    match std::env::var("BEAT_THIS_RESAMPLER").as_deref() {
        Ok("soxr") => soxr_resample(input, source_rate),
        Ok("rubato") => rubato_resample(input, source_rate),
        Ok("auto") | Err(_) => soxr_resample(input, source_rate).or_else(|cause| {
            eprintln!("warning: SoXR unavailable ({cause}), falling back to Rubato; numerical parity with PyTorch is not guaranteed");
            rubato_resample(input, source_rate)
        }),
        Ok(name) => Err(format!("invalid BEAT_THIS_RESAMPLER='{name}'; expected auto, soxr, or rubato")),
    }
}

fn mel_from_hz(hz: f32) -> f32 {
    if hz < 1000.0 {
        hz / (200.0 / 3.0)
    } else {
        15.0 + (hz / 1000.0).ln() / (6.4f32.ln() / 27.0)
    }
}

fn hz_from_mel(mel: f32) -> f32 {
    if mel < 15.0 {
        mel * (200.0 / 3.0)
    } else {
        1000.0 * ((mel - 15.0) * (6.4f32.ln() / 27.0)).exp()
    }
}

fn filters() -> Vec<f32> {
    let low = mel_from_hz(30.0);
    let high = mel_from_hz(11000.0);
    let pts: Vec<f32> = (0..(MEL_BINS + 2))
        .map(|i| hz_from_mel(low + (high - low) * i as f32 / (MEL_BINS + 1) as f32))
        .collect();
    let mut filters = vec![0f32; (FFT_SIZE / 2 + 1) * MEL_BINS];
    for freq in 0..(FFT_SIZE / 2 + 1) {
        let hz = freq as f32 * SAMPLE_RATE as f32 / FFT_SIZE as f32;
        for mel in 0..MEL_BINS {
            let rising = (hz - pts[mel]) / (pts[mel + 1] - pts[mel]);
            let falling = (pts[mel + 2] - hz) / (pts[mel + 2] - pts[mel + 1]);
            filters[freq * MEL_BINS + mel] = rising.min(falling).max(0.0);
        }
    }
    filters
}

fn reflect_index(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let upper = length as isize - 1;
    let cycle = 2 * upper;
    let mut pos = index.rem_euclid(cycle);
    if pos > upper {
        pos = cycle - pos;
    }
    pos as usize
}

pub fn log_mel(pcm: &[f32]) -> Result<Vec<[f32; MEL_BINS]>, String> {
    if pcm.is_empty() {
        return Err("no samples for spectrogram".into());
    }
    let n_frames = pcm.len() / HOP + 1;
    let fb = filters();
    let window: Vec<f32> = (0..FFT_SIZE)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32).cos()))
        .collect();
    let mut fft = FftPlanner::<f32>::new();
    let forward: Arc<dyn rustfft::Fft<f32>> = fft.plan_fft_forward(FFT_SIZE);
    let mut work = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
    let mut output = Vec::with_capacity(n_frames);
    for frame in 0..n_frames {
        let centered = (frame * HOP) as isize;
        for (i, dst) in work.iter_mut().enumerate() {
            let src = reflect_index(centered + i as isize - (FFT_SIZE / 2) as isize, pcm.len());
            *dst = Complex32::new(pcm[src] * window[i], 0.0);
        }
        forward.process(&mut work);
        let mut mel = [0f32; MEL_BINS];
        for freq in 0..=FFT_SIZE / 2 {
            let magnitude = work[freq].norm() / (FFT_SIZE as f32).sqrt();
            let row = &fb[freq * MEL_BINS..(freq + 1) * MEL_BINS];
            for i in 0..MEL_BINS {
                mel[i] += magnitude * row[i];
            }
        }
        mel.iter_mut().for_each(|x| *x = (1000.0 * *x).ln_1p());
        output.push(mel);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mel_has_expected_geometry_and_finite_values() {
        let pcm: Vec<f32> = (0..44_100)
            .map(|i| ((i as f32) * 2.0 * std::f32::consts::PI * 440.0 / 22050.0).sin() * 0.3)
            .collect();
        let spec = log_mel(&pcm).unwrap();
        assert_eq!(spec.len(), 101);
        assert!(
            spec.iter()
                .flat_map(|frame| frame.iter())
                .all(|x| x.is_finite() && *x >= 0.0)
        );
    }
    #[test]
    fn reflect_padding_is_symmetric() {
        assert_eq!(reflect_index(-1, 5), 1);
        assert_eq!(reflect_index(5, 5), 3);
    }
}
