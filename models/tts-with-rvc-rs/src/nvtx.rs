//! Optional NVTX ranges for Stage 102 CUDA profiling; disabled by default.

#[cfg(windows)]
mod native {
    use std::ffi::{c_char, c_void, CString};
    use std::sync::OnceLock;

    type Push = unsafe extern "C" fn(*const c_char) -> i32;
    type Pop = unsafe extern "C" fn() -> i32;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    struct Api { push: Push, pop: Pop }
    static API: OnceLock<Option<Api>> = OnceLock::new();

    fn api() -> Option<&'static Api> {
        API.get_or_init(|| {
            // The DLL must be discoverable through the Windows DLL search path.
            let module = unsafe { LoadLibraryA(c"nvToolsExt64_1.dll".as_ptr()) };
            if module.is_null() { return None; }
            let push = unsafe { GetProcAddress(module, c"nvtxRangePushA".as_ptr()) };
            let pop = unsafe { GetProcAddress(module, c"nvtxRangePop".as_ptr()) };
            if push.is_null() || pop.is_null() { return None; }
            Some(Api {
                push: unsafe { std::mem::transmute::<*mut c_void, Push>(push) },
                pop: unsafe { std::mem::transmute::<*mut c_void, Pop>(pop) },
            })
        }).as_ref()
    }

    pub(super) fn available() -> bool { api().is_some() }
    pub(super) fn push(name: &str) -> bool {
        let Some(api) = api() else { return false; };
        let Ok(name) = CString::new(name) else { return false; };
        unsafe { (api.push)(name.as_ptr()); }
        true
    }
    pub(super) fn pop() {
        if let Some(api) = api() { unsafe { (api.pop)(); } }
    }
}

#[cfg(not(windows))]
mod native {
    pub(super) fn available() -> bool { false }
    pub(super) fn push(_: &str) -> bool { false }
    pub(super) fn pop() {}
}

pub fn available() -> bool { native::available() }

pub struct Range(bool);

impl Range {
    pub fn new(name: &str) -> Self {
        let active = std::env::var("RVC_NVTX").as_deref() == Ok("1") && native::push(name);
        Self(active)
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        if self.0 { native::pop(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_is_safe() {
        let _guard = Range::new("stage102_unit_test");
    }
}
