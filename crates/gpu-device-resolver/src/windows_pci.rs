use burn::tensor::PciAddress;
use windows_sys::{
    Wdk::Graphics::Direct3D::{
        D3DKMT_ADAPTERADDRESS, D3DKMT_CLOSEADAPTER, D3DKMT_OPENADAPTERFROMLUID,
        D3DKMT_QUERYADAPTERINFO, D3DKMTCloseAdapter, D3DKMTOpenAdapterFromLuid,
        D3DKMTQueryAdapterInfo, KMTQAITYPE_ADAPTERADDRESS,
    },
    Win32::Foundation::LUID,
};

pub(super) fn pci_from_luid(bytes: [u8; 8]) -> Option<PciAddress> {
    let luid = LUID {
        LowPart: u32::from_le_bytes(bytes[..4].try_into().ok()?),
        HighPart: i32::from_le_bytes(bytes[4..].try_into().ok()?),
    };
    let mut open = D3DKMT_OPENADAPTERFROMLUID {
        AdapterLuid: luid,
        hAdapter: 0,
    };
    if unsafe { D3DKMTOpenAdapterFromLuid(&mut open) } < 0 {
        return None;
    }
    let mut address = D3DKMT_ADAPTERADDRESS::default();
    let mut query = D3DKMT_QUERYADAPTERINFO {
        hAdapter: open.hAdapter,
        Type: KMTQAITYPE_ADAPTERADDRESS,
        pPrivateDriverData: (&mut address as *mut D3DKMT_ADAPTERADDRESS).cast(),
        PrivateDriverDataSize: std::mem::size_of::<D3DKMT_ADAPTERADDRESS>() as u32,
    };
    let query_result = unsafe { D3DKMTQueryAdapterInfo(&mut query) };
    let _ = unsafe {
        D3DKMTCloseAdapter(&D3DKMT_CLOSEADAPTER {
            hAdapter: open.hAdapter,
        })
    };
    if query_result < 0
        || (address.BusNumber == 0 && address.DeviceNumber == 0 && address.FunctionNumber == 0)
        || address.DeviceNumber > 31
        || address.FunctionNumber > 7
        || address.BusNumber > 255
    {
        return None;
    }
    Some(PciAddress {
        domain: 0,
        bus: address.BusNumber as u8,
        device: address.DeviceNumber as u8,
        function: address.FunctionNumber as u8,
    })
}
