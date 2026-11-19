use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FishError {
    InvalidConfig(&'static str),
    InvalidPosition { position: usize, capacity: usize },
    InvalidTransition(&'static str),
    InvalidShape(&'static str),
    InvalidSampling(&'static str),
    Unsupported(&'static str),
    Backend(String),
}

impl Display for FishError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid model config: {msg}"),
            Self::InvalidPosition { position, capacity } => {
                write!(f, "position {position} exceeds KV capacity {capacity}")
            }
            Self::InvalidTransition(msg) => write!(f, "invalid state transition: {msg}"),
            Self::InvalidShape(msg) => write!(f, "invalid tensor shape: {msg}"),
            Self::InvalidSampling(msg) => write!(f, "invalid sampling parameters: {msg}"),
            Self::Backend(msg) => write!(f, "backend error: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl Error for FishError {}

pub type Result<T> = std::result::Result<T, FishError>;
