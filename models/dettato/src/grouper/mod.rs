pub mod pipeline;
pub mod net;
mod lane;
mod layers;
pub mod front;
pub mod cluster;
pub mod decode;
pub mod instruments;
pub mod render;
pub mod naming;

pub use decode::{GroupConfig, GroupPredictions, InstrumentGroup, WindowPrediction, decode_groups};
pub use instruments::{INSTRUMENTS, Instrument};
pub use render::render_tracks;
pub use naming::TrackHead;

pub use net::{NoteNet, NoteNetConfig, WindowInput, WindowOutput};
pub use pipeline::Grouper;
