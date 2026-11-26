#[cfg(feature = "cpu")]
#[test]
fn transcription_profile_reports_separate_times() {
    use std::{path::PathBuf, process::Command};
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backend");
    let output = Command::new(env!("CARGO_BIN_EXE_gigaam-rs"))
        .arg("transcribe")
        .arg(fixtures.join("ctc"))
        .arg(fixtures.join("tone.wav"))
        .args(["--device", "cpu", "--profile"])
        .output()
        .expect("running the native inference executable");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.lines().any(|line| line.starts_with("[0] ")));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("[profile] model_load_s="));
    for key in [
        "model_load_s",
        "audio_decode_s",
        "transcribe_s",
        "audio_s",
        "rtf",
    ] {
        let (_, value) = stderr
            .split_whitespace()
            .find_map(|word| word.split_once('=').filter(|(name, _)| *name == key))
            .unwrap_or_else(|| panic!("missing timing field {key}: {stderr}"));
        let parsed: f64 = value.parse().unwrap();
        assert!(
            parsed.is_finite() && parsed >= 0.0,
            "invalid {key}: {parsed}"
        );
    }
}
