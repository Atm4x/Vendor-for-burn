use serde::Deserialize;
use thiserror::Error;

use fish_speech_core::audio::{
    FIREFLY_DOWNSAMPLE, FIREFLY_GROUPS, FIREFLY_LEVELS, HIFIGAN_KERNELS, HIFIGAN_UPSAMPLE,
    HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE,
};

#[derive(Debug, Error)]
pub enum FireflyConfigError {
    #[error("Firefly JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported Firefly configuration: {0}")]
    Unsupported(String),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FireflyConfig {
    #[serde(rename = "_target_")]
    target: String,
    spec_transform: LogMelConfig,
    backbone: ConvNeXtConfig,
    head: HiFiGanConfig,
    quantizer: FsqConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct LogMelConfig {
    #[serde(rename = "_target_")]
    target: String,
    sample_rate: usize,
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConvNeXtConfig {
    #[serde(rename = "_target_")]
    target: String,
    input_channels: usize,
    depths: [usize; 4],
    dims: [usize; 4],
    drop_path_rate: f32,
    kernel_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HiFiGanConfig {
    #[serde(rename = "_target_")]
    target: String,
    hop_length: usize,
    upsample_rates: [usize; 5],
    upsample_kernel_sizes: [usize; 5],
    resblock_kernel_sizes: [usize; 3],
    resblock_dilation_sizes: [[usize; 3]; 3],
    num_mels: usize,
    upsample_initial_channel: usize,
    pre_conv_kernel_size: usize,
    post_conv_kernel_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FsqConfig {
    #[serde(rename = "_target_")]
    target: String,
    input_dim: usize,
    n_groups: usize,
    n_codebooks: usize,
    levels: [usize; 4],
    downsample_factor: [usize; 2],
}

fn expect<T: PartialEq + std::fmt::Debug>(
    key: &str,
    actual: T,
    expected: T,
) -> Result<(), FireflyConfigError> {
    if actual != expected {
        return Err(FireflyConfigError::Unsupported(format!(
            "{key}: expected {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

impl FireflyConfig {
    pub fn parse_supported(json: &[u8]) -> Result<Self, FireflyConfigError> {
        let result: Self = serde_json::from_slice(json)?;
        result.validate_supported()?;
        Ok(result)
    }

    pub fn validate_supported(&self) -> Result<(), FireflyConfigError> {
        expect(
            "target",
            self.target.as_str(),
            "fish_speech_lib.fish_speech.models.vqgan.modules.firefly.FireflyArchitecture",
        )?;
        expect(
            "spec_transform.target",
            self.spec_transform.target.as_str(),
            "fish_speech_lib.fish_speech.utils.spectrogram.LogMelSpectrogram",
        )?;
        expect(
            "spec_transform.sample_rate",
            self.spec_transform.sample_rate,
            SAMPLE_RATE,
        )?;
        expect("spec_transform.n_mels", self.spec_transform.n_mels, N_MELS)?;
        expect("spec_transform.n_fft", self.spec_transform.n_fft, N_FFT)?;
        expect(
            "spec_transform.hop_length",
            self.spec_transform.hop_length,
            HOP_LENGTH,
        )?;
        expect(
            "spec_transform.win_length",
            self.spec_transform.win_length,
            N_FFT,
        )?;
        expect(
            "backbone.target",
            self.backbone.target.as_str(),
            "fish_speech_lib.fish_speech.models.vqgan.modules.firefly.ConvNeXtEncoder",
        )?;
        expect(
            "backbone.input_channels",
            self.backbone.input_channels,
            N_MELS,
        )?;
        expect("backbone.depths", self.backbone.depths, [3, 3, 9, 3])?;
        expect("backbone.dims", self.backbone.dims, [128, 256, 384, 512])?;
        expect("backbone.drop_path_rate", self.backbone.drop_path_rate, 0.2)?;
        expect("backbone.kernel_size", self.backbone.kernel_size, 7)?;
        expect(
            "head.target",
            self.head.target.as_str(),
            "fish_speech_lib.fish_speech.models.vqgan.modules.firefly.HiFiGANGenerator",
        )?;
        expect("head.hop_length", self.head.hop_length, HOP_LENGTH)?;
        expect(
            "head.upsample_rates",
            self.head.upsample_rates,
            HIFIGAN_UPSAMPLE,
        )?;
        expect(
            "head.upsample_kernel_sizes",
            self.head.upsample_kernel_sizes,
            HIFIGAN_KERNELS,
        )?;
        expect(
            "head.resblock_kernel_sizes",
            self.head.resblock_kernel_sizes,
            [3, 7, 11],
        )?;
        expect(
            "head.resblock_dilation_sizes",
            self.head.resblock_dilation_sizes,
            [[1, 3, 5]; 3],
        )?;
        expect("head.num_mels", self.head.num_mels, 512)?;
        expect(
            "head.upsample_initial_channel",
            self.head.upsample_initial_channel,
            512,
        )?;
        expect(
            "head.pre_conv_kernel_size",
            self.head.pre_conv_kernel_size,
            13,
        )?;
        expect(
            "head.post_conv_kernel_size",
            self.head.post_conv_kernel_size,
            13,
        )?;
        expect(
            "quantizer.target",
            self.quantizer.target.as_str(),
            "fish_speech_lib.fish_speech.models.vqgan.modules.fsq.DownsampleFiniteScalarQuantize",
        )?;
        expect("quantizer.input_dim", self.quantizer.input_dim, 512)?;
        expect(
            "quantizer.n_groups",
            self.quantizer.n_groups,
            FIREFLY_GROUPS,
        )?;
        expect("quantizer.n_codebooks", self.quantizer.n_codebooks, 1)?;
        expect("quantizer.levels", self.quantizer.levels, FIREFLY_LEVELS)?;
        expect(
            "quantizer.downsample_factor",
            self.quantizer.downsample_factor,
            FIREFLY_DOWNSAMPLE,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = include_str!("../tests/fixtures/firefly_1_5.json");

    #[test]
    fn python_fork_firefly_config_matches_fixed_rust_graph() {
        FireflyConfig::parse_supported(SOURCE.as_bytes()).unwrap();
    }

    #[test]
    fn python_parameter_shapes_match_rust_encoder_and_decoder() {
        use fish_speech_core::firefly_ir::FireflyGraph;
        let mut rust = FireflyGraph::reference_encode(1, 8192).unwrap().weights;
        for (key, value) in FireflyGraph::waveform_decode(1, 8, 8).unwrap().weights {
            if let Some(old) = rust.insert(key.clone(), value.clone()) {
                assert_eq!(old, value, "duplicate incompatible weight: {key}");
            }
        }
        let golden = include_str!("../tests/fixtures/firefly_python_weight_shapes.tsv");
        let mut reference = std::collections::BTreeMap::new();
        for line in golden.lines() {
            let (key, raw) = line.split_once('\t').unwrap();
            let shape = raw
                .split(',')
                .map(|d| d.parse::<usize>().unwrap())
                .collect::<Vec<_>>();
            assert!(
                reference.insert(key, shape).is_none(),
                "duplicate Python key: {key}"
            );
        }
        assert_eq!(reference.len(), 450);
        assert_eq!(rust.len(), reference.len());
        for (key, shape) in reference {
            assert_eq!(&rust[key].shape, &shape, "Python mismatch for {key}");
        }
    }

    #[test]
    fn rejects_unsupported_firefly_variants_instead_of_silent_shape_mismatch() {
        for (key, value) in [
            ("spec_transform", "n_mels"),
            ("backbone", "depths"),
            ("head", "upsample_kernel_sizes"),
            ("quantizer", "n_groups"),
        ] {
            let mut config: serde_json::Value = serde_json::from_str(SOURCE).unwrap();
            config[key][value] = serde_json::json!(1);
            assert!(FireflyConfig::parse_supported(&serde_json::to_vec(&config).unwrap()).is_err());
        }
    }

    #[test]
    fn rejects_missing_unknown_and_untrusted_python_expressions() {
        let mut config: serde_json::Value = serde_json::from_str(SOURCE).unwrap();
        config["quantizer"]["downsample_dims"] = serde_json::json!([32, 64]);
        assert!(FireflyConfig::parse_supported(&serde_json::to_vec(&config).unwrap()).is_err());
        assert!(FireflyConfig::parse_supported(b"__import__('os').system('bad')").is_err());
        assert!(FireflyConfig::parse_supported(b"{}").is_err());
    }
}
