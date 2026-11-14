use burn::tensor::Device;
use serde_json::{Value, json};
use std::{cell::Cell, sync::atomic::{AtomicU64, Ordering}, time::{Instant, SystemTime, UNIX_EPOCH}};

static NEXT_BATCH: AtomicU64 = AtomicU64::new(1);
thread_local! { static BATCH: Cell<u64> = const { Cell::new(0) }; static PHASE: Cell<&'static str> = const { Cell::new("setup") }; }

pub fn set_phase(value: &'static str) { if enabled() { PHASE.set(value); } }

pub fn enabled() -> bool {
    std::env::var_os("RVC_DIAGNOSTIC_TRACE").is_some_and(|value| value == "1")
}

pub fn batch_start(batch: usize, samples: usize, frames: usize) -> (u64, Instant) {
    let id = NEXT_BATCH.fetch_add(1, Ordering::Relaxed);
    if enabled() {
        BATCH.set(id);
        emit("batch_begin", json!({"batch": batch, "input_samples_per_item": samples, "input_frames": frames, "input_hz": 16000}));
    }
    (id, Instant::now())
}

pub fn batch_end(id: u64, start: Instant) {
    if enabled() {
        emit("batch_end", json!({"batch_id": id, "wall_ms": ms(start)}));
        BATCH.set(0);
    }
}

pub fn ms(start: Instant) -> f64 { start.elapsed().as_secs_f64() * 1000.0 }

fn unix_ms() -> u128 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() }

pub fn emit(name: &str, data: Value) {
    if !enabled() { return; }
    let id = BATCH.get();
    eprintln!("RVC_TRACE {}", json!({
        "type": "event", "event": name, "batch_id": id, "phase": PHASE.get(), "unix_ms": unix_ms(), "data": data,
    }));
}

pub fn measure(device: &Device, name: &str, started: Instant, details: Value) {
    if !enabled() { return; }
    let dispatched_ms = ms(started);
    let sync_start = Instant::now();
    let sync_result = device.sync();
    let sync_ms = ms(sync_start);
    let id = BATCH.get();
    eprintln!("RVC_TRACE {}", json!({
        "type": "span", "name": name, "batch_id": id, "phase": PHASE.get(), "unix_ms": unix_ms(),
        "total_ms": ms(started), "enqueue_and_cpu_ms": dispatched_ms,
        "sync_ms": sync_ms, "details": details,
        "measurement": "wall_clock_with_explicit_device_sync; includes diagnostic overhead",
        "sync_error": sync_result.as_ref().err().map(|e| format!("{e:?}")),
    }));
    if let Err(error) = sync_result {
        panic!("diagnostic GPU synchronization at {name} failed: {error:?}");
    }
}
