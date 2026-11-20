pub mod cfm;
pub mod checkpoint;
pub mod cross_lingual;
#[cfg(any(
    feature = "cubecl-cpu",
    feature = "cubecl-cuda",
    feature = "cubecl-wgpu"
))]
pub mod dit;
#[cfg(any(
    feature = "cubecl-cpu",
    feature = "cubecl-cuda",
    feature = "cubecl-wgpu"
))]
pub mod dit_input;
#[cfg(any(
    feature = "cubecl-cpu",
    feature = "cubecl-cuda",
    feature = "cubecl-wgpu"
))]
pub mod dit_model;
pub mod fft;
pub mod hyphenation;
pub mod mel;
pub mod model;
pub mod pinyin;
#[cfg(any(
    feature = "cubecl-cpu",
    feature = "cubecl-cuda",
    feature = "cubecl-wgpu"
))]
pub mod pipeline;
pub mod reference;
pub mod resample;
#[cfg(any(
    feature = "cubecl-cpu",
    feature = "cubecl-cuda",
    feature = "cubecl-wgpu"
))]
pub mod speed_predictor;
pub mod standard_chunking;
pub mod text_frontend;
pub mod vocab;
#[cfg(any(
    feature = "cubecl-cpu",
    feature = "cubecl-cuda",
    feature = "cubecl-wgpu"
))]
pub mod vocos;
pub mod wav;

pub mod backend;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum F5Error {
    InvalidConfiguration(&'static str),
    InvalidShape(&'static str),
    InvalidCheckpoint(String),
    ModelExecution(String),
}

impl core::fmt::Display for F5Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfiguration(message) | Self::InvalidShape(message) => {
                write!(f, "{message}")
            }
            Self::InvalidCheckpoint(message) | Self::ModelExecution(message) => {
                f.write_str(message)
            }
        }
    }
}

impl std::error::Error for F5Error {}

pub type Result<T> = std::result::Result<T, F5Error>;
