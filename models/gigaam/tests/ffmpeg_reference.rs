use anyhow::Result;
use gigaam_rs::audio;
use std::{path::Path, process::Command};

#[test]
fn ffmpeg_float32_matches_reference_and_supports_compressed_audio() -> Result<()> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("FFmpeg unavailable, skipping external decoder test");
        return Ok(());
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backend/tone.wav");
    let pcm = audio::load_audio_ffmpeg_16k(&path)?;
    assert_eq!(pcm.len(), 6400);
    assert!(pcm.iter().all(|x| x.is_finite()));
    let reference = Command::new("ffmpeg")
        .args(["-nostdin", "-threads", "0", "-i"])
        .arg(&path)
        .args(["-f", "f32le", "-ac", "1", "-acodec", "pcm_f32le", "-ar", "16000", "-"])
        .output()?;
    assert!(reference.status.success());
    assert_eq!(pcm.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(), reference.stdout);

    let dir = std::env::temp_dir().join(format!("gigaam-ffmpeg-reference-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let quiet = dir.join("quiet.wav");
    let mut writer = hound::WavWriter::create(&quiet, hound::WavSpec {
        channels: 1, sample_rate: 16000, bits_per_sample: 32, sample_format: hound::SampleFormat::Float,
    })?;
    let samples = [1e-6f32, -3e-6, 0.1234567, -0.2345678];
    for sample in samples { writer.write_sample(sample)?; }
    writer.finalize()?;
    assert_eq!(audio::load_audio_ffmpeg_16k(&quiet)?, samples);
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
