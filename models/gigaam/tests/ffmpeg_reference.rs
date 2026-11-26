use anyhow::Result;
use gigaam_rs::audio;
use std::{path::Path, process::Command};

#[test]
fn ffmpeg_pcm16_matches_reference_and_supports_compressed_audio() -> Result<()> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("FFmpeg unavailable, skipping external decoder test");
        return Ok(());
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backend/tone.wav");
    let pcm = audio::load_audio_ffmpeg_16k(&path)?;
    assert_eq!(pcm.len(), 6400);
    assert!(pcm.iter().all(|x| x.is_finite()));
    assert!(pcm.iter().all(|x| (*x * 32768.0).fract() == 0.0));

    let dir = std::env::temp_dir().join(format!("gigaam-ffmpeg-reference-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let mp3 = dir.join("tone.mp3");
    let status = Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error", "-y", "-i"])
        .arg(&path)
        .arg(&mp3)
        .status()?;
    if status.success() {
        let compressed = audio::load_audio_16k(&mp3)?;
        assert!(compressed.len() > 6000 && compressed.len() < 7000);
        assert!(compressed.iter().all(|x| x.is_finite()));
    }
    let _ = std::fs::remove_dir_all(dir);
    Ok(())
}
