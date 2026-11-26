pub mod accent;
pub mod omograph;
pub mod stress;
pub mod yo;

use std::sync::OnceLock;

pub(crate) fn environment() -> ort::Result<&'static ort::environment::Environment> {
    static ENVIRONMENT: OnceLock<Result<ort::environment::Environment, String>> = OnceLock::new();
    ENVIRONMENT
        .get_or_init(|| ort::init().build().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| ort::Error::new(error.clone()))
}
