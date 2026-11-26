/// Deterministic, random-access Gaussian field keyed by absolute output position.
///
/// The source is intentionally distinct from the legacy per-chunk SplitMix64
/// generator and from PyTorch's RNG. Its invariant is partition independence,
/// not bitwise equivalence to another random-number implementation.
#[derive(Clone, Copy, Debug)]
pub struct GlobalGaussianField {
    seed: u64,
    total_frames: usize,
    hop: usize,
}

impl GlobalGaussianField {
    const LATENT_DOMAIN: u64 = 0x532d_4c41_5445_4e54;
    const SOURCE_DOMAIN: u64 = 0x532d_4558_4349_5445;

    pub fn new(seed: u64, total_frames: usize, hop: usize) -> Result<Self, String> {
        if total_frames == 0 || hop == 0 || total_frames.checked_mul(hop).is_none()
            || total_frames * hop > i64::MAX as usize
        {
            return Err("global Gaussian noise requires positive bounded length and hop".into());
        }
        Ok(Self { seed, total_frames, hop })
    }

    fn splitmix64(mut value: u64) -> u64 {
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn uniform(seed: u64) -> f64 {
        ((Self::splitmix64(seed) >> 11) as f64 + 0.5) * (1.0 / (1_u64 << 53) as f64)
    }

    fn normal(&self, domain: u64, channel: usize, position: usize) -> f32 {
        let pair = position / 2;
        let key = Self::splitmix64(
            self.seed ^ domain ^ (channel as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
                ^ (pair as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let radius = (-2.0 * Self::uniform(key).ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * Self::uniform(key ^ 0xa076_1d64_78bd_642f);
        (radius * if position % 2 == 0 { angle.cos() } else { angle.sin() }) as f32
    }

    fn reflect(position: i64, length: usize) -> usize {
        if length <= 1 { return 0; }
        let period = 2_i64 * (length as i64 - 1);
        let position = position.rem_euclid(period);
        if position >= length as i64 { (period - position) as usize } else { position as usize }
    }

    pub fn latent_at(&self, channel: usize, absolute_frame: i64) -> f32 {
        self.normal(Self::LATENT_DOMAIN, channel, Self::reflect(absolute_frame, self.total_frames))
    }

    pub fn excitation_at(&self, absolute_sample: i64) -> f32 {
        self.normal(Self::SOURCE_DOMAIN, 0, Self::reflect(absolute_sample, self.total_frames * self.hop))
    }

    pub fn fill_latent(&self, channels: usize, frames: usize, start_frame: i64, scale: f32) -> Result<Vec<f32>, String> {
        if channels == 0 || frames == 0 || !scale.is_finite() || scale < 0.0
            || channels.checked_mul(frames).is_none()
        {
            return Err("invalid global Gaussian latent geometry or scale".into());
        }
        let mut output = Vec::with_capacity(channels * frames);
        for channel in 0..channels {
            for frame in 0..frames {
                let absolute = start_frame.checked_add(frame as i64).ok_or("latent position overflow")?;
                output.push(self.latent_at(channel, absolute) * scale);
            }
        }
        Ok(output)
    }

    pub fn fill_excitation(&self, frames: usize, start_frame: i64) -> Result<Vec<f32>, String> {
        let count = frames.checked_mul(self.hop).ok_or("excitation noise length overflow")?;
        if frames == 0 || count > i64::MAX as usize {
            return Err("invalid global Gaussian excitation geometry".into());
        }
        let first = start_frame.checked_mul(self.hop as i64).ok_or("excitation start overflow")?;
        let mut output = Vec::with_capacity(count);
        for index in 0..count {
            let position = first.checked_add(index as i64).ok_or("excitation position overflow")?;
            output.push(self.excitation_at(position));
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::GlobalGaussianField;

    #[test]
    fn excitation_is_exactly_partition_invariant() {
        let field = GlobalGaussianField::new(9001, 1000, 400).unwrap();
        let full = field.fill_excitation(720, 0).unwrap();
        for sizes in [[100, 200, 420], [300, 300, 120], [1, 318, 401]] {
            let mut joined = Vec::new();
            let mut start = 0;
            for size in sizes {
                joined.extend(field.fill_excitation(size, start).unwrap());
                start += size as i64;
            }
            assert_eq!(full, joined);
        }
    }

    #[test]
    fn latent_is_exactly_partition_invariant_in_channel_major_order() {
        let field = GlobalGaussianField::new(51, 1000, 400).unwrap();
        let full = field.fill_latent(64, 880, 0, 0.66666).unwrap();
        for channels in [1, 8, 64] {
            for channel in 0..channels {
                for frame in 0..880 {
                    assert_eq!(full[channel * 880 + frame], field.latent_at(channel, frame as i64) * 0.66666);
                }
            }
        }
        let a = field.fill_latent(4, 390, 27, 1.0).unwrap();
        let b = field.fill_latent(4, 390, 27, 1.0).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn overlapping_context_and_reflection_match_global_indices() {
        let field = GlobalGaussianField::new(6, 900, 400).unwrap();
        for &(start, frames) in &[(0, 384), (300, 384), (517, 384), (899, 85)] {
            let first = start - 40;
            let source = field.fill_excitation(frames, first).unwrap();
            let latent = field.fill_latent(8, frames, first, 1.0).unwrap();
            for offset in [0, 3, 40, frames - 1] {
                for channel in 0..8 {
                    assert_eq!(latent[channel * frames + offset], field.latent_at(channel, first + offset as i64));
                }
                assert_eq!(source[offset * 400], field.excitation_at((first + offset as i64) * 400));
            }
        }
    }

    #[test]
    fn streams_and_seed_are_domain_separated() {
        let a = GlobalGaussianField::new(10, 1000, 400).unwrap();
        let b = GlobalGaussianField::new(11, 1000, 400).unwrap();
        assert_ne!(a.latent_at(0, 123), a.excitation_at(123));
        assert_ne!(a.latent_at(0, 123), a.latent_at(1, 123));
        assert_ne!(a.latent_at(0, 123), b.latent_at(0, 123));
    }

    #[test]
    fn gaussian_moments_are_reasonable() {
        let a = GlobalGaussianField::new(235, 1000, 400).unwrap();
        let samples = a.fill_excitation(400, 0).unwrap();
        let n = samples.len() as f64;
        let mean = samples.iter().map(|x| *x as f64).sum::<f64>() / n;
        let variance = samples.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 0.015, "Gaussian mean {mean}");
        assert!((variance - 1.0).abs() < 0.025, "Gaussian variance {variance}");
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        assert!(GlobalGaussianField::new(1, 0, 400).is_err());
        assert!(GlobalGaussianField::new(1, 100, 0).is_err());
        let field = GlobalGaussianField::new(1, 10, 400).unwrap();
        assert!(field.fill_latent(10, 1, 0, f32::NAN).is_err());
        assert!(field.fill_excitation(0, 0).is_err());
        assert!(field.fill_excitation(usize::MAX, 0).is_err());
    }
}
