#![cfg(feature = "tensorrt-mock")]

use std::{
    cell::{Cell, RefCell},
    panic::{AssertUnwindSafe, catch_unwind},
};

use fish_speech_vendor::trtx_engine::{EngineError, with_resident_engine_sessions};

#[test]
fn reuses_two_contexts_across_prefill_decode_and_syncs_once() {
    let logger = trtx::Logger::stderr().unwrap();
    let syncs = Cell::new(0);
    let ids = RefCell::new(Vec::new());
    let result = with_resident_engine_sessions(
        &logger,
        &[("slow-prefill", &[1, 2][..]), ("slow-decode", &[3, 4][..])],
        |contexts| {
            assert_eq!(contexts.len(), 2);
            for key in ["slow-prefill", "slow-decode", "slow-decode", "slow-decode"] {
                ids.borrow_mut()
                    .push((key, contexts.get_mut(key).unwrap() as *mut _ as usize));
            }
            assert_eq!(ids.borrow()[1].1, ids.borrow()[2].1);
            assert_eq!(ids.borrow()[2].1, ids.borrow()[3].1);
            assert_eq!(syncs.get(), 0);
            Ok(4)
        },
        || {
            syncs.set(syncs.get() + 1);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(result, 4);
    assert_eq!(syncs.get(), 1);
}

#[test]
fn error_and_panic_flush_before_teardown() {
    let logger = trtx::Logger::stderr().unwrap();
    let synced = Cell::new(false);
    let result: Result<(), _> = with_resident_engine_sessions(
        &logger,
        &[("slow", &[1][..])],
        |_| Err(EngineError::InvalidBindings("failed prefill".into())),
        || {
            synced.set(true);
            Ok(())
        },
    );
    assert!(result.is_err());
    assert!(synced.get());
    synced.set(false);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _: Result<(), _> = with_resident_engine_sessions(
            &logger,
            &[("slow", &[1][..])],
            |_| panic!("injected failure"),
            || {
                synced.set(true);
                Ok(())
            },
        );
    }));
    assert!(panic.is_err());
    assert!(synced.get());
}

#[test]
fn malformed_resident_plan_rejected_before_runtime() {
    let logger = trtx::Logger::stderr().unwrap();
    for images in [
        vec![],
        vec![("slow", &[][..])],
        vec![("slow", &[1][..]), ("slow", &[2][..])],
        vec![("", &[1][..])],
    ] {
        let result: Result<(), _> = with_resident_engine_sessions(
            &logger,
            &images,
            |_| panic!("unexpected TRT execution"),
            || panic!("unexpected stream synchronization"),
        );
        assert!(matches!(result, Err(EngineError::InvalidPlan(_))));
    }
}

#[test]
fn user_managed_context_path_retains_owned_memory_until_sync() {
    use fish_speech_vendor::trtx_engine::with_resident_engine_sessions_managed;
    let logger = trtx::Logger::stderr().unwrap();
    let syncs = Cell::new(0);
    let allocations = Cell::new(0);
    let result = with_resident_engine_sessions_managed(
        &logger,
        &[("prefill", &[1, 2][..]), ("decode", &[3, 4][..])],
        None,
        |_bytes| {
            allocations.set(allocations.get() + 1);
            Err::<(fish_speech_vendor::trtx_engine::DeviceRegion, ()), _>(
                EngineError::InvalidBindings("mock context should require zero workspace".into()),
            )
        },
        |contexts| {
            assert_eq!(contexts.len(), 2);
            assert_eq!(syncs.get(), 0);
            Ok(2)
        },
        || {
            syncs.set(syncs.get() + 1);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(result, 2);
    assert_eq!(syncs.get(), 1);
    assert_eq!(allocations.get(), 0);
}
