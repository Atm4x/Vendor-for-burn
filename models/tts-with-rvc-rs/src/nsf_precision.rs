#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NsfConvPrecision {
    F32,
    MixedF16 { min_channels: usize },
}

impl NsfConvPrecision {
    pub(crate) fn from_values(
        flag: Option<&str>,
        min_channels: Option<&str>,
    ) -> Result<Self, String> {
        if flag != Some("1") {
            return Ok(Self::F32);
        }
        let min_channels = match min_channels {
            Some(value) => value.parse::<usize>().map_err(|_| {
                "RVC_NSF_RESBLOCK_FP16_MIN_CHANNELS must be a positive integer".to_owned()
            })?,
            None => 64,
        };
        if min_channels == 0 {
            return Err("RVC_NSF_RESBLOCK_FP16_MIN_CHANNELS must be positive".to_owned());
        }
        Ok(Self::MixedF16 { min_channels })
    }

    pub(crate) fn from_env() -> Result<Self, String> {
        let enabled = std::env::var("RVC_NSF_RESBLOCK_FP16").ok();
        let threshold = std::env::var("RVC_NSF_RESBLOCK_FP16_MIN_CHANNELS").ok();
        Self::from_values(enabled.as_deref(), threshold.as_deref())
    }

    pub(crate) fn applies_to(&self, layer_name: &str, channels: usize) -> bool {
        match self {
            Self::F32 => false,
            Self::MixedF16 { min_channels } => {
                layer_name.starts_with("dec.resblocks.") && channels >= *min_channels
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_does_not_change_production() {
        let policy = NsfConvPrecision::from_values(None, Some("32")).unwrap();
        assert_eq!(policy, NsfConvPrecision::F32);
        assert!(!policy.applies_to("dec.resblocks.0.convs1.0", 256));
    }

    #[test]
    fn shape_threshold_selects_only_resblocks() {
        let policy = NsfConvPrecision::from_values(Some("1"), None).unwrap();
        assert!(policy.applies_to("dec.resblocks.3.convs2.2", 128));
        assert!(policy.applies_to("dec.resblocks.6.convs2.1", 64));
        assert!(!policy.applies_to("dec.resblocks.9.convs2.1", 32));
        assert!(!policy.applies_to("dec.conv_pre", 256));
        assert!(!policy.applies_to("dec.ups.0", 256));
        let all = NsfConvPrecision::from_values(Some("1"), Some("32")).unwrap();
        assert!(all.applies_to("dec.resblocks.9.convs2.1", 32));
        let major = NsfConvPrecision::from_values(Some("1"), Some("128")).unwrap();
        assert!(!major.applies_to("dec.resblocks.6.convs2.1", 64));
    }

    #[test]
    fn invalid_threshold_is_rejected_explicitly() {
        assert!(NsfConvPrecision::from_values(Some("1"), Some("zero")).is_err());
        assert!(NsfConvPrecision::from_values(Some("1"), Some("0")).is_err());
        assert!(NsfConvPrecision::from_values(Some("0"), Some("0")).is_ok());
    }
}
