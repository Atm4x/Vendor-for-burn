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

/// Configure external native runtime DLLs before Burn or CubeCL initializes.
///
/// `directory` adjusts the Windows native dependency search path; explicit DLLs
/// are loaded by absolute path and verified against the OS module table.
/// This does not change the compiled Burn backend (CUDA, WGPU, etc.).
pub fn configure_native_libraries(
    backend: &str,
    directory: Option<&std::path::Path>,
    libraries: &[std::path::PathBuf],
) -> Result<(), String> {
    if directory.is_none() && libraries.is_empty() {
        return Ok(());
    }
    if !matches!(backend, "cuda" | "wgpu") {
        return Err("external GPU runtime DLLs require cuda or wgpu backend".into());
    }
    let mut explicit = libraries.to_vec();
    let mut search_dir = None;
    if let Some(path) = directory {
        let path = validate_runtime_path(path)?;
        if path.is_file() {
            explicit.insert(0, path);
        } else {
            search_dir = Some(path);
        }
    }
    let mut used = std::collections::HashMap::new();
    for path in &mut explicit {
        *path = validate_runtime_path(path)?;
        if !path.is_file()
            || !path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("dll"))
        {
            return Err(format!(
                "native runtime library must be a DLL file: {}",
                path.display()
            ));
        }
        let filename = path
            .file_name()
            .ok_or("DLL has no file name")?
            .to_string_lossy()
            .to_ascii_lowercase();
        if let Some(previous) = used.insert(filename.clone(), path.clone()) {
            if previous != *path {
                return Err(format!(
                    "conflicting DLL paths for {filename}: {} and {}",
                    previous.display(),
                    path.display()
                ));
            }
            return Err(format!("duplicate runtime DLL: {filename}"));
        }
    }
    #[cfg(windows)]
    return windows_native::load(search_dir.as_deref(), &explicit);
    #[cfg(not(windows))]
    {
        let _ = (search_dir, explicit);
        Err("external DLL paths are supported on Windows only".into())
    }
}

fn validate_runtime_path(path: &std::path::Path) -> Result<std::path::PathBuf, String> {
    if !path.is_absolute() {
        return Err("runtime path must be absolute".into());
    }
    let meta = std::fs::symlink_metadata(path)
        .map_err(|error| format!("invalid runtime path {}: {error}", path.display()))?;
    if meta.file_type().is_symlink() || !(meta.is_file() || meta.is_dir()) {
        return Err("runtime path must be an existing, non-symlink file or directory".into());
    }
    path.canonicalize()
        .map_err(|error| format!("runtime path {}: {error}", path.display()))
}

#[cfg(windows)]
mod windows_native {
    use std::{
        ffi::{OsStr, OsString},
        os::windows::ffi::{OsStrExt, OsStringExt},
        path::{Path, PathBuf},
    };
    use windows_sys::Win32::System::LibraryLoader::{
        AddDllDirectory, GetModuleFileNameW, GetModuleHandleW, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_USER_DIRS, LoadLibraryExW,
        SetDefaultDllDirectories,
    };

    fn wide(value: &OsStr) -> Vec<u16> {
        value.encode_wide().chain(Some(0)).collect()
    }

    fn module_path(handle: windows_sys::Win32::Foundation::HMODULE) -> Result<PathBuf, String> {
        let mut buf = vec![0_u16; 32768];
        let len =
            unsafe { GetModuleFileNameW(handle, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        if len == 0 || len >= buf.len() {
            return Err(format!(
                "GetModuleFileNameW failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        PathBuf::from(OsString::from_wide(&buf[..len]))
            .canonicalize()
            .map_err(|e| format!("cannot canonicalize loaded DLL: {e}"))
    }

    fn assert_module_path(
        handle: windows_sys::Win32::Foundation::HMODULE,
        expected: &Path,
    ) -> Result<(), String> {
        let actual = module_path(handle)?;
        if !actual
            .to_string_lossy()
            .eq_ignore_ascii_case(&expected.to_string_lossy())
        {
            return Err(format!(
                "runtime DLL path conflict: requested {}, loaded {}",
                expected.display(),
                actual.display()
            ));
        }
        Ok(())
    }

    pub(super) fn load(directory: Option<&Path>, libraries: &[PathBuf]) -> Result<(), String> {
        if let Some(directory) = directory {
            let dir_wide = wide(directory.as_os_str());
            if unsafe { AddDllDirectory(dir_wide.as_ptr()) }.is_null() {
                return Err(format!(
                    "cannot register native DLL directory: {}",
                    std::io::Error::last_os_error()
                ));
            }
            if unsafe {
                SetDefaultDllDirectories(
                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS,
                )
            } == 0
            {
                return Err(format!(
                    "cannot enable registered DLL directories: {}",
                    std::io::Error::last_os_error()
                ));
            }
        }
        for library in libraries {
            let name = wide(library.file_name().ok_or("runtime DLL has no file name")?);
            let already_loaded = unsafe { GetModuleHandleW(name.as_ptr()) };
            if !already_loaded.is_null() {
                assert_module_path(already_loaded, library)?;
            }
            let path = wide(library.as_os_str());
            let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
            let module = unsafe { LoadLibraryExW(path.as_ptr(), std::ptr::null_mut(), flags) };
            if module.is_null() {
                return Err(format!(
                    "cannot load runtime DLL {}: {}",
                    library.display(),
                    std::io::Error::last_os_error()
                ));
            }
            assert_module_path(module, library)?;
            let name_loaded = unsafe { GetModuleHandleW(name.as_ptr()) };
            if name_loaded.is_null() {
                return Err(format!(
                    "runtime DLL {} did not register by module name",
                    library.display()
                ));
            }
            assert_module_path(name_loaded, library)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod native_runtime_tests {
    use super::*;

    #[test]
    fn absent_external_libraries_preserve_default_runtime() {
        assert!(configure_native_libraries("cuda", None, &[]).is_ok());
        assert!(configure_native_libraries("cpu", None, &[]).is_ok());
    }

    #[test]
    fn cpu_never_accepts_external_gpu_dlls() {
        assert!(
            configure_native_libraries("cpu", Some(std::path::Path::new("C:/runtime")), &[])
                .is_err()
        );
    }

    #[test]
    fn invalid_runtime_library_path_is_rejected() {
        assert!(configure_native_libraries("cuda", None, &["relative.dll".into()]).is_err());
    }

    #[test]
    fn duplicate_library_names_cannot_silently_override_each_other() {
        let root = std::env::temp_dir().join(format!("rvc-native-loader-{}", uuid::Uuid::new_v4()));
        let first = root.join("one");
        let second = root.join("two");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        let a = first.join("nvrtc64_120_0.dll");
        let b = second.join("nvrtc64_120_0.dll");
        std::fs::write(&a, b"test").unwrap();
        std::fs::write(&b, b"test").unwrap();
        let result = configure_native_libraries("cuda", None, &[a, b]);
        assert!(result.unwrap_err().contains("conflicting DLL paths"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn explicit_windows_system_dll_is_loaded_from_requested_path() {
        let system_root = std::env::var_os("SystemRoot").expect("Windows SystemRoot");
        let dll = std::path::PathBuf::from(system_root)
            .join("System32")
            .join("kernel32.dll");
        assert!(dll.is_file());
        configure_native_libraries("cuda", None, &[dll]).expect("absolute runtime DLL load");
    }
}
