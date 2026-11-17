use crate::error::{FishError, Result};

pub const SAMPLE_RATE: usize = 44_100;
pub const N_FFT: usize = 2048;
pub const HOP_LENGTH: usize = 512;
pub const N_MELS: usize = 160;
pub const FIREFLY_GROUPS: usize = 8;
pub const FIREFLY_LEVELS: [usize; 4] = [8, 5, 5, 5];
pub const FIREFLY_DOWNSAMPLE: [usize; 2] = [2, 2];
pub const HIFIGAN_UPSAMPLE: [usize; 5] = [8, 8, 2, 2, 2];
pub const HIFIGAN_KERNELS: [usize; 5] = [16, 16, 4, 4, 4];

pub fn fish_conv_output_len(
    len: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> Result<usize> {
    if len == 0 || kernel == 0 || stride == 0 || dilation == 0 {
        return Err(FishError::InvalidShape("nonpositive convolution dimension"));
    }
    let effective = (kernel - 1)
        .checked_mul(dilation)
        .and_then(|v| v.checked_add(1))
        .ok_or(FishError::InvalidShape("kernel overflow"))?;
    if effective < stride {
        return Err(FishError::InvalidShape(
            "FishConvNet needs effective kernel >= stride",
        ));
    }
    let padded = len
        .checked_add(stride - 1)
        .ok_or(FishError::InvalidShape("length overflow"))?;
    Ok(padded / stride)
}

pub fn fish_conv_padding(
    len: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> Result<(usize, usize)> {
    let effective = (kernel - 1)
        .checked_mul(dilation)
        .and_then(|v| v.checked_add(1))
        .ok_or(FishError::InvalidShape("kernel overflow"))?;
    if effective < stride {
        return Err(FishError::InvalidShape(
            "FishConvNet needs effective kernel >= stride",
        ));
    }
    let target = fish_conv_output_len(len, kernel, stride, dilation)?
        .checked_mul(stride)
        .ok_or(FishError::InvalidShape("length overflow"))?;
    Ok((effective - stride, target - len))
}

pub fn fish_transconv_output_len(
    len: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> Result<usize> {
    if len == 0 || kernel == 0 || stride == 0 || dilation == 0 {
        return Err(FishError::InvalidShape("nonpositive transconv dimension"));
    }
    if kernel < stride {
        return Err(FishError::InvalidShape(
            "FishTransConvNet unpad requires kernel >= stride",
        ));
    }
    let effective = (kernel - 1)
        .checked_mul(dilation)
        .and_then(|x| x.checked_add(1))
        .ok_or(FishError::InvalidShape("transconv kernel overflow"))?;
    let raw = (len - 1)
        .checked_mul(stride)
        .and_then(|x| x.checked_add(effective))
        .ok_or(FishError::InvalidShape("transconv overflow"))?;
    Ok(raw - (kernel - stride))
}

pub fn firefly_encoded_frames(mel_frames: usize) -> Result<usize> {
    let mut n = mel_frames;
    for &s in &FIREFLY_DOWNSAMPLE {
        n = fish_conv_output_len(n, s, s, 1)?;
    }
    Ok(n)
}

pub fn firefly_decoded_samples(code_frames: usize) -> Result<usize> {
    let mut n = code_frames;
    for &s in FIREFLY_DOWNSAMPLE.iter().rev() {
        n = fish_transconv_output_len(n, s, s, 1)?;
    }
    for (&s, &k) in HIFIGAN_UPSAMPLE.iter().zip(HIFIGAN_KERNELS.iter()) {
        n = fish_transconv_output_len(n, k, s, 1)?;
    }
    Ok(n)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireflyFrameLengths {
    pub input_audio_samples: usize,
    pub mel_frames: usize,
    /// Length of the quantizer output tensor after two stride-2 convolutions.
    pub allocated_code_frames: usize,
    /// Original Python uses (audio_lengths // hop_length) // downsample_factor.
    pub valid_code_frames: usize,
}

pub fn firefly_reference_lengths(input_audio_samples: usize) -> Result<FireflyFrameLengths> {
    // Original LinearSpectrogram uses F.pad(..., (768, 768), mode="reflect")
    // before an uncentered STFT. Reflection requires input length > 768.
    if input_audio_samples <= (N_FFT - HOP_LENGTH) / 2 {
        return Err(FishError::InvalidShape(
            "Fish reflect-padding requires audio length > 768",
        ));
    }
    let mel_frames = input_audio_samples / HOP_LENGTH;
    let allocated_code_frames = firefly_encoded_frames(mel_frames)?;
    let downsample = FIREFLY_DOWNSAMPLE.iter().product::<usize>();
    let valid_code_frames = mel_frames / downsample;
    Ok(FireflyFrameLengths {
        input_audio_samples,
        mel_frames,
        allocated_code_frames,
        valid_code_frames,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireflyDecodeLengths {
    pub allocated_code_frames: usize,
    pub valid_code_frames: usize,
    pub allocated_mel_frames: usize,
    pub valid_mel_frames: usize,
    pub allocated_audio_samples: usize,
    pub valid_audio_samples: usize,
}

pub fn firefly_decode_lengths(
    allocated_code_frames: usize,
    valid_code_frames: usize,
) -> Result<FireflyDecodeLengths> {
    if allocated_code_frames == 0 || valid_code_frames > allocated_code_frames {
        return Err(FishError::InvalidShape(
            "invalid quantizer length or valid-frame count",
        ));
    }
    let downsample = FIREFLY_DOWNSAMPLE.iter().product::<usize>();
    let allocated_mel_frames = allocated_code_frames
        .checked_mul(downsample)
        .ok_or(FishError::InvalidShape("Firefly mel frame count overflow"))?;
    let valid_mel_frames = valid_code_frames
        .checked_mul(downsample)
        .ok_or(FishError::InvalidShape("Firefly valid mel frames overflow"))?;
    let allocated_audio_samples = firefly_decoded_samples(allocated_code_frames)?;
    let valid_audio_samples = valid_mel_frames
        .checked_mul(HOP_LENGTH)
        .ok_or(FishError::InvalidShape("Firefly output samples overflow"))?;
    if valid_audio_samples > allocated_audio_samples {
        return Err(FishError::InvalidShape(
            "Firefly decoded sample length mismatch",
        ));
    }
    Ok(FireflyDecodeLengths {
        allocated_code_frames,
        valid_code_frames,
        allocated_mel_frames,
        valid_mel_frames,
        allocated_audio_samples,
        valid_audio_samples,
    })
}

pub fn mask_generated_audio_in_place(
    samples: &mut [f32],
    batch: usize,
    lengths: &FireflyDecodeLengths,
) -> Result<()> {
    let expected = batch
        .checked_mul(lengths.allocated_audio_samples)
        .ok_or(FishError::InvalidShape("Firefly audio mask size overflow"))?;
    if batch == 0 || samples.len() != expected {
        return Err(FishError::InvalidShape(
            "Firefly audio mask wrong input shape",
        ));
    }
    for frames in samples.chunks_exact_mut(lengths.allocated_audio_samples) {
        frames[lengths.valid_audio_samples..].fill(0.0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encoder_hop_and_fsqs() {
        assert_eq!(FIREFLY_GROUPS, 8);
        assert_eq!(FIREFLY_LEVELS.iter().product::<usize>(), 1000);
        assert_eq!(HIFIGAN_UPSAMPLE.iter().product::<usize>(), HOP_LENGTH);
    }
    #[test]
    fn conv_same_when_stride_one() {
        for l in 1..100 {
            assert_eq!(fish_conv_output_len(l, 7, 1, 1).unwrap(), l);
        }
    }
    #[test]
    fn conv_ceil_for_downsampling() {
        assert_eq!(fish_conv_output_len(7, 2, 2, 1).unwrap(), 4);
        assert_eq!(fish_conv_padding(7, 2, 2, 1).unwrap(), (0, 1));
    }
    #[test]
    fn conv_dilated_uses_effective_kernel() {
        assert_eq!(fish_conv_padding(9, 3, 1, 5).unwrap(), (10, 0));
    }
    #[test]
    fn transconv_output_is_stride_times_length() {
        for l in [1, 7, 64] {
            assert_eq!(fish_transconv_output_len(l, 16, 8, 1).unwrap(), l * 8);
        }
    }
    #[test]
    fn encoded_frames_ceil_div_by_four() {
        assert_eq!(firefly_encoded_frames(7).unwrap(), 2);
        assert_eq!(firefly_encoded_frames(8).unwrap(), 2);
    }
    #[test]
    fn decoded_codes_have_2048_samples() {
        assert_eq!(firefly_decoded_samples(11).unwrap(), 11 * 4 * HOP_LENGTH);
    }
    #[test]
    fn encoder_allocated_length_is_not_always_valid_length() {
        let l = firefly_reference_lengths(4 * HOP_LENGTH + 1).unwrap();
        assert_eq!(l.mel_frames, 4);
        assert_eq!(l.allocated_code_frames, 1);
        assert_eq!(l.valid_code_frames, 1);
        let l = firefly_reference_lengths(7 * HOP_LENGTH + 1).unwrap();
        assert_eq!(l.mel_frames, 7);
        assert_eq!(l.allocated_code_frames, 2);
        assert_eq!(l.valid_code_frames, 1);
        let l = firefly_reference_lengths(769).unwrap();
        assert_eq!(
            (l.mel_frames, l.allocated_code_frames, l.valid_code_frames),
            (1, 1, 0)
        );
    }
    #[test]
    fn reflect_padding_short_audio_is_invalid_in_original() {
        assert!(firefly_reference_lengths(768).is_err());
        assert!(firefly_reference_lengths(0).is_err());
        assert!(firefly_reference_lengths(769).is_ok());
    }
    #[test]
    fn mask_uses_valid_feature_frames_not_allocated_frames() {
        let length = firefly_decode_lengths(3, 2).unwrap();
        assert_eq!(length.allocated_mel_frames, 12);
        assert_eq!(length.valid_mel_frames, 8);
        assert_eq!(length.allocated_audio_samples, 3 * 2048);
        assert_eq!(length.valid_audio_samples, 2 * 2048);
        let mut audio = vec![1.0f32; length.allocated_audio_samples * 2];
        mask_generated_audio_in_place(&mut audio, 2, &length).unwrap();
        for b in 0..2 {
            let samples = &audio
                [b * length.allocated_audio_samples..(b + 1) * length.allocated_audio_samples];
            assert!(samples[..length.valid_audio_samples]
                .iter()
                .all(|&x| x == 1.0));
            assert!(samples[length.valid_audio_samples..]
                .iter()
                .all(|&x| x == 0.0));
        }
    }
    #[test]
    fn mask_rejects_wrong_batch_and_overrun() {
        assert!(firefly_decode_lengths(2, 3).is_err());
        assert!(firefly_decode_lengths(0, 0).is_err());
        let len = firefly_decode_lengths(2, 1).unwrap();
        assert!(mask_generated_audio_in_place(&mut [0.0; 4], 1, &len).is_err());
    }
    #[test]
    fn transconv_overflow_is_error_not_panic() {
        assert!(fish_transconv_output_len(usize::MAX, 4, 2, 1).is_err());
        assert!(fish_transconv_output_len(2, 4, 2, usize::MAX).is_err());
    }
    #[test]
    fn input_validation() {
        assert!(fish_conv_output_len(0, 3, 1, 1).is_err());
        assert!(fish_transconv_output_len(3, 1, 4, 1).is_err());
    }
}
