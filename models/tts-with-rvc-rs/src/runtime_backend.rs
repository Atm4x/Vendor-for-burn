use burn::tensor::Device;

pub fn selected_device(name: &str) -> Result<Device, String> {
    match name {
        #[cfg(feature = "cpu")]
        "cpu" => Ok(Device::cpu()),
        #[cfg(all(not(feature = "cpu"), feature = "flex"))]
        "cpu" => Ok(Device::flex()),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::cuda(0)),
        #[cfg(feature = "wgpu")]
        "wgpu" => Ok(Device::wgpu(Default::default())),
        other if matches!(other, "cpu" | "cuda" | "wgpu") => Err(format!(
            "backend {other} is not compiled into this executable; no fallback"
        )),
        other => Err(format!(
            "unsupported backend {other}; expected cpu, cuda or wgpu"
        )),
    }
}
