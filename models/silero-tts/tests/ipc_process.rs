use serde_json::{Value, json};
use std::{io::{BufRead, BufReader, Write}, path::{Path, PathBuf}, process::{Child, ChildStdin, ChildStdout, Command, Stdio}, sync::atomic::{AtomicUsize, Ordering}};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Harness {
    root: PathBuf,
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    next: usize,
}
impl Harness {
    fn new(checkpoint: Option<&Path>) -> Self {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!("silero-ipc-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir(&root).unwrap();
        let fake = root.join("bad.pt");
        std::fs::write(&fake, b"not a model").unwrap();
        let checkpoint = checkpoint.unwrap_or(&fake).canonicalize().unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_silero-tts"))
            .arg("--ipc-stdio").args(["--backend", "flex", "--checkpoint"]).arg(checkpoint)
            .arg("--input-dir").arg(&root).arg("--output-dir").arg(&root)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().unwrap();
        let input = child.stdin.take().unwrap();
        let output = BufReader::new(child.stdout.take().unwrap());
        Self { root, child, input, output, next: 0 }
    }
    fn request(&mut self, command: &str, payload: Value, ok: bool) -> Value {
        self.next += 1;
        let id = format!("00000000-0000-4000-8000-{:012x}", self.next);
        writeln!(self.input, "{}", json!({"version":1,"id":id,"command":command,"payload":payload})).unwrap();
        self.input.flush().unwrap();
        let mut line = String::new();
        assert!(self.output.read_line(&mut line).unwrap() > 0, "worker closed stdout");
        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["id"], id);
        assert_eq!(response["version"], 1);
        assert_eq!(response["ok"], ok, "{response}");
        if ok { assert!(response["error"].is_null()); response["result"].clone() }
        else { assert!(response["result"].is_null()); assert!(response["error"].is_string()); response["error"].clone() }
    }
    fn shutdown(&mut self) { self.request("shutdown", json!({}), true); assert!(self.child.wait().unwrap().success()); }
}
impl Drop for Harness {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); let _ = std::fs::remove_dir_all(&self.root); }
}

#[cfg(feature = "reference-flex")]
#[test]
fn shared_transport_rejects_invalid_state_payload_and_corrupt_model() {
    let mut w = Harness::new(None);
    assert_eq!(w.request("capabilities", json!({}), true)["service"], "tts");
    assert_eq!(w.request("status", json!({}), true)["phase"], "started");
    w.request("warmup", json!({}), false);
    w.request("load", json!({"checkpoint":"other.pt"}), false);
    w.request("load", json!({}), false);
    w.request("unload", json!({}), false);
    assert_eq!(w.request("status", json!({}), true)["phase"], "started");
    w.shutdown();
}

#[cfg(feature = "reference-flex")]
#[test]
#[ignore = "requires official checkpoints; set SILERO_V3_EN_CHECKPOINT and SILERO_V5_RU_CHECKPOINT"]
fn shared_transport_real_checkpoint_lifecycle() {
    for (key, variant, text, speaker, count) in [
        ("SILERO_V3_EN_CHECKPOINT", "v3_en", "Hello, world!", "en_0", 7800),
        ("SILERO_V5_RU_CHECKPOINT", "v5_ru", "Привет.", "aidar", 5800),
    ] {
        let path = std::env::var(key).expect(key);
        let mut w = Harness::new(Some(Path::new(&path)));
        let input = json!({"silero_version":1,"speaker":speaker,"sample_rate":8000,"input":{"kind":"text","text":text}});
        w.request("infer", input.clone(), false);
        assert_eq!(w.request("load", json!({}), true)["model_variant"], variant);
        w.request("load", json!({}), false);
        w.request("infer", input.clone(), false);
        assert_eq!(w.request("warmup", json!({}), true)["phase"], "ready");
        let first = w.request("infer", input.clone(), true);
        let second = w.request("infer", input.clone(), true);
        assert_eq!(first["sample_count"], count);
        assert_eq!(second["sample_count"], count);
        assert_ne!(first["audio"]["file_name"], second["audio"]["file_name"]);
        let a: native_worker_ipc::Snapshot = serde_json::from_value(first["audio"].clone()).unwrap();
        let b: native_worker_ipc::Snapshot = serde_json::from_value(second["audio"].clone()).unwrap();
        let bytes = a.read(&w.root).unwrap();
        assert_eq!(bytes, b.read(&w.root).unwrap());
        let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(reader.spec().sample_rate, 8000);
        assert_eq!(reader.duration(), count);
        assert!(reader.samples::<i16>().any(|sample| sample.unwrap() != 0));
        let mut bad = input; bad["speaker"] = json!("not-a-speaker");
        w.request("infer", bad, false);
        assert_eq!(w.request("status", json!({}), true)["successful_inferences"], 2);
        assert_eq!(w.request("unload", json!({}), true)["phase"], "started");
        w.shutdown();
    }
}
