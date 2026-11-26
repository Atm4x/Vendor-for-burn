//! Select a Burn compute device by physical GPU identity, independently of CUDA or WGPU indexes.

use burn::tensor::{Device, PciAddress, PhysicalDevice};
use std::{fmt, str::FromStr};

#[cfg(feature = "cuda")]
use burn::tensor::DeviceType;
#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
use burn::tensor::DeviceType;

#[cfg(windows)]
mod windows_pci;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Wgpu,
}

impl FromStr for Backend {
    type Err = ResolveError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cuda" => Ok(Self::Cuda),
            "wgpu" => Ok(Self::Wgpu),
            _ => Err(ResolveError::InvalidBackend(value.to_owned())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuId {
    Pci(PciAddress),
    /// Windows-local LUID; never persist across restarts.
    WindowsLuid([u8; 8]),
}

impl fmt::Display for GpuId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pci(value) => write!(f, "pci:{value}"),
            Self::WindowsLuid(bytes) => {
                write!(f, "luid:")?;
                for byte in bytes {
                    write!(f, "{byte:02x}")?;
                }
                Ok(())
            }
        }
    }
}

impl FromStr for GpuId {
    type Err = ResolveError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if let Some(pci) = value.strip_prefix("pci:") {
            let address: PciAddress = pci
                .parse()
                .map_err(|_| ResolveError::InvalidId(value.to_owned()))?;
            if address.device > 31 || address.function > 7 {
                return Err(ResolveError::InvalidId(value.to_owned()));
            }
            return Ok(Self::Pci(address));
        }
        if let Some(luid) = value.strip_prefix("luid:") {
            if luid.len() != 16 || !luid.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(ResolveError::InvalidId(value.to_owned()));
            }
            let mut bytes = [0u8; 8];
            for (i, item) in bytes.iter_mut().enumerate() {
                *item = u8::from_str_radix(&luid[i * 2..i * 2 + 2], 16)
                    .map_err(|_| ResolveError::InvalidId(value.to_owned()))?;
            }
            return Ok(Self::WindowsLuid(bytes));
        }
        Err(ResolveError::InvalidId(value.to_owned()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolveError {
    InvalidId(String),
    InvalidBackend(String),
    BackendNotCompiled(Backend),
    EnumerationFailed(Backend),
    DeviceNotFound {
        backend: Backend,
        id: GpuId,
    },
    AmbiguousDevice {
        backend: Backend,
        id: GpuId,
        count: usize,
    },
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidId(value) => write!(
                f,
                "invalid physical GPU ID {value:?}; expected pci:0000:01:00.0 or luid:<16 hex digits>"
            ),
            Self::InvalidBackend(value) => write!(f, "unsupported GPU backend {value:?}"),
            Self::BackendNotCompiled(value) => {
                write!(f, "GPU backend {value:?} was not compiled into the SDK")
            }
            Self::EnumerationFailed(value) => write!(
                f,
                "cannot enumerate {value:?} devices; check the installed driver"
            ),
            Self::DeviceNotFound { backend, id } => {
                write!(f, "physical GPU {id} not found on {backend:?}; no fallback")
            }
            Self::AmbiguousDevice { backend, id, count } => write!(
                f,
                "physical GPU {id} matches {count} {backend:?} devices; refusing ambiguous selection"
            ),
        }
    }
}

impl std::error::Error for ResolveError {}

#[derive(Debug, Clone)]
pub struct DiscoveredGpu {
    pub id: Option<GpuId>,
    pub name: String,
    pub pci_address: Option<PciAddress>,
    pub windows_luid: Option<[u8; 8]>,
}

#[derive(Debug, Clone)]
pub struct ResolvedGpu {
    pub device: Device,
    pub info: DiscoveredGpu,
}

fn gpu_pci(physical: &PhysicalDevice) -> Option<PciAddress> {
    #[cfg(windows)]
    if let Some(luid) = physical.luid {
        let os_pci = windows_pci::pci_from_luid(luid.bytes());
        if let (Some(runtime_pci), Some(os_pci)) = (physical.pci_address, os_pci) {
            if runtime_pci != os_pci {
                return None;
            }
        }
        return physical.pci_address.or(os_pci);
    }
    physical.pci_address
}

fn info(name: String, physical: &PhysicalDevice) -> DiscoveredGpu {
    let pci = gpu_pci(physical);
    let luid = physical.luid.map(|value| value.bytes());
    DiscoveredGpu {
        id: pci.map(GpuId::Pci).or_else(|| luid.map(GpuId::WindowsLuid)),
        name,
        pci_address: pci,
        windows_luid: luid,
    }
}

fn matches_gpu(id: GpuId, physical: &PhysicalDevice) -> bool {
    match id {
        GpuId::Pci(expected) => gpu_pci(physical) == Some(expected),
        GpuId::WindowsLuid(expected) => {
            physical.luid.is_some_and(|value| value.bytes() == expected)
        }
    }
}

fn choose_unique<T>(
    backend: Backend,
    id: GpuId,
    candidates: impl IntoIterator<Item = (T, PhysicalDevice)>,
) -> Result<T, ResolveError> {
    let mut matches = candidates
        .into_iter()
        .filter_map(|(value, physical)| matches_gpu(id, &physical).then_some(value));
    let first = matches.next();
    let extra = matches.count();
    match (first, extra) {
        (Some(value), 0) => Ok(value),
        (Some(_), _) => Err(ResolveError::AmbiguousDevice {
            backend,
            id,
            count: extra + 1,
        }),
        (None, _) => Err(ResolveError::DeviceNotFound { backend, id }),
    }
}

fn enumerate_backend(backend: Backend) -> Result<Vec<Device>, ResolveError> {
    match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            std::panic::catch_unwind(|| Device::enumerate(DeviceType::Cuda).into_vec())
                .map_err(|_| ResolveError::EnumerationFailed(backend))
        }
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => {
            std::panic::catch_unwind(|| Device::enumerate(DeviceType::Wgpu).into_vec())
                .map_err(|_| ResolveError::EnumerationFailed(backend))
        }
        #[cfg(not(feature = "cuda"))]
        Backend::Cuda => Err(ResolveError::BackendNotCompiled(backend)),
        #[cfg(not(feature = "wgpu"))]
        Backend::Wgpu => Err(ResolveError::BackendNotCompiled(backend)),
    }
}

/// Enumerate the devices exposed by one compiled compute backend.
/// The PCI ID comes from the runtime or, on Windows, from D3DKMT using the runtime's LUID.
/// Entries with no reported physical identifier cannot be selected by physical GPU ID.
/// Device creation happens only when the result is subsequently passed to `resolve`.
pub fn enumerate(backend: Backend) -> Result<Vec<DiscoveredGpu>, ResolveError> {
    let devices = enumerate_backend(backend)?;
    Ok(devices
        .into_iter()
        .filter_map(|device| {
            let identity = device.identity()?;
            let physical = identity.physical?;
            Some(info(identity.name, &physical))
        })
        .collect())
}

/// Resolve exactly one physical GPU through the chosen backend. Never selects index 0 as fallback.
pub fn resolve(backend: Backend, id: GpuId) -> Result<ResolvedGpu, ResolveError> {
    let mut entries = Vec::new();
    for device in enumerate_backend(backend)? {
        if let Some(identity) = device.identity() {
            if let Some(physical) = identity.physical {
                entries.push(((device, identity.name), physical));
            }
        }
    }
    let (device, name) = choose_unique(backend, id, entries)?;
    let physical = device
        .identity()
        .and_then(|entry| entry.physical)
        .ok_or(ResolveError::DeviceNotFound { backend, id })?;
    if !matches_gpu(id, &physical) {
        return Err(ResolveError::DeviceNotFound { backend, id });
    }
    Ok(ResolvedGpu {
        device,
        info: info(name, &physical),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(pci: &str) -> PhysicalDevice {
        let mut device = PhysicalDevice::default();
        device.pci_address = Some(pci.parse().unwrap());
        device
    }

    #[test]
    fn pci_and_luid_ids_round_trip() {
        for value in [
            "pci:0000:01:00.0",
            "pci:0000:07:00.1",
            "luid:7856341201000000",
        ] {
            let id: GpuId = value.parse().unwrap();
            assert_eq!(id.to_string(), value);
        }
        for bad in [
            "",
            "cuda:1",
            "gpu0",
            "pci:bad",
            "pci:0000:ff:ff.8",
            "luid:zzz",
            "luid:1234",
        ] {
            assert!(bad.parse::<GpuId>().is_err(), "{bad}");
        }
    }

    #[test]
    fn two_identical_gpu_names_select_by_pci_only() {
        let selected = choose_unique(
            Backend::Wgpu,
            "pci:0000:02:00.0".parse().unwrap(),
            [
                ("RTX 4060 A", card("0000:01:00.0")),
                ("RTX 4060 B", card("0000:02:00.0")),
            ],
        )
        .unwrap();
        assert_eq!(selected, "RTX 4060 B");
    }

    #[test]
    fn missing_card_never_falls_back() {
        let error = choose_unique(
            Backend::Cuda,
            "pci:0000:03:00.0".parse().unwrap(),
            [(0u32, card("0000:01:00.0"))],
        )
        .unwrap_err();
        assert!(matches!(error, ResolveError::DeviceNotFound { .. }));
    }

    #[test]
    fn duplicate_matching_device_is_rejected() {
        let error = choose_unique(
            Backend::Wgpu,
            "pci:0000:01:00.0".parse().unwrap(),
            [(0u32, card("0000:01:00.0")), (1u32, card("0000:01:00.0"))],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ResolveError::AmbiguousDevice { count: 2, .. }
        ));
    }

    #[test]
    fn luid_only_match_is_exact() {
        use burn::tensor::AdapterLuid;
        let mut first = PhysicalDevice::default();
        first.luid = Some(AdapterLuid::new([1; 8]));
        let mut second = PhysicalDevice::default();
        second.luid = Some(AdapterLuid::new([2; 8]));
        assert_eq!(
            choose_unique(
                Backend::Wgpu,
                GpuId::WindowsLuid([2; 8]),
                [(10, first), (20, second)]
            )
            .unwrap(),
            20
        );
    }
}
