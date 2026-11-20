pub mod features;
pub mod repair;
pub mod candidates;
pub mod embedding;
pub mod model;
mod pipeline;

pub use features::{EVIDENCE_FEATURES, MAP_FEATURES, HIDDEN_CHANNELS, evidence_from_taps};
pub use repair::{RepairCandidate, RepairConfig, RepairInfo, RepairKind, apply_repair, input_notes};
