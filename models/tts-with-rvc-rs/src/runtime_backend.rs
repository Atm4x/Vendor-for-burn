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

/// Explicit physical GPU selection, independent of CUDA/WGPU enumeration indexes.
/// Without an ID the historical backend default remains unchanged.
pub fn selected_physical_device(name: &str, gpu_id: Option<&str>) -> Result<Device, String> {
    let Some(raw_id) = gpu_id else {
        return selected_device(name);
    };
    let backend = name
        .parse::<gpu_device_resolver::Backend>()
        .map_err(|error| error.to_string())?;
    let id = raw_id
        .parse::<gpu_device_resolver::GpuId>()
        .map_err(|error| error.to_string())?;
    gpu_device_resolver::resolve(backend, id)
        .map(|resolved| resolved.device)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_selection_never_falls_back_to_cpu() {
        assert!(selected_physical_device("cpu", Some("pci:0000:01:00.0")).is_err());
        assert!(selected_physical_device("cuda", Some("gpu:0")).is_err());
    }
}
