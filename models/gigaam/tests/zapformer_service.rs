#![cfg(feature = "flex")]

use anyhow::{Context, Result};
use burn::tensor::Device;
use gigaam_rs::{
    AsrOptions, GigaAm, ModelKind,
    audio::SpeechRegion,
    service::{GigaAmService, Segmentation, TranscribeOptions},
    zapformer::{HeadSelection, HeadTranscriptions, language::LanguageTag},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::{Path, PathBuf}, process::{Command, Stdio}};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!("gigaam-zap-service-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

fn signal(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 * 0.067).sin() + (i as f32 * 0.101).cos()) * 0.1).collect()
}

#[test]
fn native_load_selects_decoder_and_language_without_mutating_model() -> Result<()> {
    let pcm = signal(5600);
    let language = LanguageTag::Tag("ru-en".into());
    let bad = LanguageTag::Tag("not-a-language".into());
    for (name, kind) in [("dual_ctc.ckpt", ModelKind::Ctc), ("dual_rnnt.ckpt", ModelKind::Rnnt)] {
        let path = fixtures().join("zapformer").join(name);
        let model = GigaAm::load(&path, Device::flex())?;
        assert!(model.is_zapformer());
        assert_eq!(model.model_kind(), kind);
        let first = model.transcribe_16k(&pcm)?;
        let HeadTranscriptions::Dual { raw, e2e } = model.transcribe_zapformer_heads_16k(&pcm, Some(&language))? else { panic!("missing dual heads") };
        for (head, expected) in [(HeadSelection::Raw, raw), (HeadSelection::E2e, e2e)] {
            let options = AsrOptions { head: Some(head), language: Some(&language) };
            assert_eq!(model.transcribe_16k_with_options(&pcm, options)?, expected[0]);
            assert_eq!(model.transcribe_with_timestamps_and_options_16k(&pcm, options)?.transcription, expected[0]);
        }
        assert_eq!(model.transcribe_16k(&pcm)?, first);
        assert!(model.transcribe_16k_with_options(&pcm, AsrOptions { head: None, language: Some(&bad) }).is_err());
        if kind.is_ctc() {
            let (logits, time) = model.infer_logits_16k(&pcm)?;
            assert!(time > 0 && logits.len() % time == 0 && logits.iter().all(|v| v.is_finite()));
        }
        let service = GigaAmService::from_loaded(model).with_batch_size(2)?;
        let regions = [SpeechRegion { start_sample: 320, end_sample: 2000 }, SpeechRegion { start_sample: 2400, end_sample: 5600 }];
        let asr = AsrOptions { head: Some(HeadSelection::Raw), language: Some(&language) };
        let result = service.transcribe_16k_with_asr(&pcm, TranscribeOptions { segmentation: Segmentation::SpeechRegions(&regions), word_timestamps: true }, asr)?;
        assert_eq!(result.segments.len(), regions.len());
        for (segment, region) in result.segments.iter().zip(regions) {
            assert_eq!(segment.start, region.start_sample as f64 / 16000.0);
            assert_eq!(segment.end, region.end_sample as f64 / 16000.0);
            let individual = service.transcribe_16k_with_asr(&pcm[region.start_sample..region.end_sample], TranscribeOptions { word_timestamps: true, ..Default::default() }, asr)?;
            assert_eq!(segment.text, individual.text);
            assert!(segment.words.as_ref().unwrap().iter().all(|word| word.start >= segment.start - 0.001 && word.end <= segment.end + 0.001));
        }
        let silent = service.transcribe_16k_with_asr(&pcm, TranscribeOptions { segmentation: Segmentation::SpeechRegions(&[]), word_timestamps: false }, asr)?;
        assert!(silent.segments.is_empty() && silent.text.is_empty());
    }
    let legacy = GigaAm::load(&fixtures().join("backend/ctc"), Device::flex())?;
    assert!(legacy.transcribe_16k_with_options(&pcm, AsrOptions { head: Some(HeadSelection::Raw), language: None }).is_err());
    Ok(())
}

#[test]
fn ssl_loads_safetensors_with_upstream_json_sidecar() -> Result<()> {
    let temp = Scratch::new()?;
    let cases: Value = serde_json::from_str(include_str!("fixtures/zapformer/cases.json"))?;
    let cfg = json!({"model_name":"v4_ssl", "encoder":cases[0]["config"], "preprocessor":{"features":8,"center":false,"n_fft":320,"win_length":320}});
    fs::write(temp.0.join("config.json"), serde_json::to_vec(&cfg)?)?;
    fs::copy(fixtures().join("zapformer/causal_swish.safetensors"), temp.0.join("v4_ssl.safetensors"))?;
    let model = GigaAm::load(&temp.0, Device::flex())?;
    assert_eq!(model.model_kind(), ModelKind::Ssl);
    assert!(model.is_zapformer());
    let pcm = signal(3200);
    let (values, frames, channels) = model.infer_embeddings_16k(&pcm)?;
    assert_eq!(channels, 8);
    assert_eq!(values.len(), frames * channels);
    assert!(values.iter().all(|x| x.is_finite()));
    let batch = model.infer_encoder_batch_16k(&[&pcm[..], &pcm[..2000]])?;
    assert_eq!(batch[0].1, frames);
    assert_eq!(batch[0].2, channels);
    assert!(batch[0].0.iter().zip(values).all(|(x,y)| (x-y).abs() < 2e-5));
    Ok(())
}

fn run_worker(model_name: &str, checkpoint: &str, legacy: bool) -> Result<Vec<Value>> {
    let temp = Scratch::new()?;
    let root = &temp.0;
    if legacy {
        for entry in fs::read_dir(fixtures().join("backend/ctc"))? {
            let entry = entry?;
            fs::copy(entry.path(), root.join(entry.file_name()))?;
        }
    } else {
        fs::copy(fixtures().join("zapformer").join(checkpoint), root.join(format!("{model_name}.ckpt")))?;
    }
    fs::copy(fixtures().join("backend/tone.wav"), root.join("tone.wav"))?;
    let bytes = fs::read(root.join("tone.wav"))?;
    let input = json!({"file_name":"tone.wav","size_bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))});
    let asr = |head: &str, language: &str| json!({"gigaam_version":1,"kind":"asr","input":input,"head":head,"language":language,"word_timestamps":true,"group_vad":false,"segments":[{"start":0.02,"end":0.12},{"start":0.2,"end":0.38}]});
    let model_id = format!("gigaam_{model_name}");
    let tasks = vec![
        ("capabilities",json!({})),
        ("load",json!({"model_id":model_id})),
        ("warmup",json!({})),
        ("infer",asr("raw","ru-en")),
        ("infer",asr("e2e","invalid")),
        ("infer",asr("e2e","ru-en")),
        ("infer",asr("raw","ru-en")),
        ("unload",json!({})),
        ("status",json!({})),
        ("shutdown",json!({})),
    ];
    let mut child = Command::new(env!("CARGO_BIN_EXE_gigaam-rs"))
        .args(["--ipc-stdio", "--model-id", &model_id, "--backend", "cpu", "--model-dir", root.to_str().context("path UTF-8")?, "--input-dir", root.to_str().context("path UTF-8")?])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    {
        let mut stdin = child.stdin.take().context("no stdin")?;
        for (i,(cmd,payload)) in tasks.iter().enumerate() {
            writeln!(stdin,"{}",json!({"version":1,"id":format!("00000000-0000-4000-8000-{i:012}"),"command":cmd,"payload":payload}))?;
        }
    }
    let output = child.wait_with_output()?;
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let replies: Vec<Value> = String::from_utf8(output.stdout)?.lines().map(serde_json::from_str).collect::<serde_json::Result<_>>()?;
    assert_eq!(replies.len(), tasks.len());
    Ok(replies)
}

#[test]
fn ipc_dual_head_lifecycle_preserves_external_vad_contract() -> Result<()> {
    for (id, checkpoint) in [("v4_ctc","dual_ctc.ckpt"),("multilingual_v2_rnnt","dual_rnnt.ckpt")] {
        let replies = run_worker(id, checkpoint, false)?;
        for (i, expected) in [true,true,true,true,false,true,true,true,true,true].into_iter().enumerate() {
            assert_eq!(replies[i]["ok"], expected, "{id} reply {i}: {}", replies[i]);
        }
        assert_eq!(replies[3]["result"], replies[6]["result"]);
        assert_eq!(replies[3]["result"]["segments"].as_array().unwrap().len(), 2);
        assert_eq!(replies[3]["result"]["segments"][0]["start"], 0.02);
        assert_eq!(replies[3]["result"]["segments"][1]["end"], 0.38);
        assert_eq!(replies[8]["result"]["phase"], "started");
    }
    let wrong = run_worker("v4_ctc", "", true)?;
    assert_eq!(wrong[1]["ok"], false);
    assert!(wrong[1]["error"].as_str().unwrap().contains("encoder architecture"));
    Ok(())
}

#[test]
fn pcm_streaming_matches_upstream_packet_by_packet() -> Result<()> {
    use gigaam_rs::{checkpoint_config::read_checkpoint_metadata, zapformer::{config::CheckpointConfig, ZapformerModel, StreamPacket, StreamTranscription}};
    fn check(actual: &StreamTranscription, expected: &Value) -> Result<()> {
        assert_eq!(actual.text, expected["text"].as_str().unwrap());
        assert_eq!(actual.token_ids, serde_json::from_value::<Vec<usize>>(expected["token_ids"].clone())?);
        assert_eq!(actual.frames, serde_json::from_value::<Vec<usize>>(expected["frames"].clone())?);
        match &actual.words {
            None => assert!(expected["words"].is_null()),
            Some(words) => {
                let reference = expected["words"].as_array().unwrap();
                assert_eq!(words.len(), reference.len());
                for (a,b) in words.iter().zip(reference) {
                    assert_eq!(a.text,b["text"].as_str().unwrap());
                    assert!((a.start-b["start"].as_f64().unwrap()).abs()<1e-9);
                    assert!((a.end-b["end"].as_f64().unwrap()).abs()<1e-9);
                }
            }
        }
        Ok(())
    }
    let reference: Value = serde_json::from_str(include_str!("fixtures/zapformer/pcm_stream_reference.json"))?;
    let packets: Vec<usize> = serde_json::from_value(reference["packets"].clone())?;
    let pcm: Vec<f32> = (0..packets.iter().sum()).map(|i:usize| ((i*1297%4096) as f32-2048.0)/16384.0).collect();
    for kind in ["ctc","rnnt"] {
        let path = fixtures().join("zapformer").join(format!("dual_{kind}.ckpt"));
        let mut cfg = CheckpointConfig::from_upstream(&read_checkpoint_metadata(&path)?.config)?;
        cfg.streaming = Some(json!({"contexts":[{"future_context_ms":160,"left_context":8},{"future_context_ms":320,"left_context":8},{"future_context_ms":640,"left_context":8}]}));
        let model = ZapformerModel::load_with_config(&path,cfg,Device::flex())?;
        for case in reference["cases"].as_array().unwrap().iter().filter(|c|c["kind"]==kind) {
            let ms = case["ms"].as_u64().unwrap() as usize;
            let timestamps = case["timestamps"].as_bool().unwrap();
            let tag = LanguageTag::Tag("ru-en".into());
            let mut stream = model.start_stream(Some(ms),Some(&tag),timestamps)?;
            let mut pos=0;
            for (&count,expected) in packets.iter().chain(std::iter::once(&0)).zip(case["packets"].as_array().unwrap()) {
                let StreamPacket::Dual {raw,e2e} = stream.push(&pcm[pos..pos+count],count==0)? else { panic!("missing dual result") };
                check(&raw,&expected["raw"]).with_context(||format!("{kind} {ms} {timestamps} raw packet at {pos}"))?;
                check(&e2e,&expected["e2e"]).with_context(||format!("{kind} {ms} {timestamps} e2e packet at {pos}"))?;
                pos+=count;
            }
            assert!(stream.finished());
            let StreamPacket::Dual {raw,e2e} = stream.flush()? else {panic!("missing dual result")};
            assert!(raw.text.is_empty() && e2e.text.is_empty() && raw.token_ids.is_empty() && e2e.token_ids.is_empty());
            assert_eq!(pos,pcm.len());
        }
        let mut first = model.start_stream(None,None,false)?;
        let mut second = model.start_stream(Some(640),None,false)?;
        assert_eq!(first.push(&pcm[..5600],false)?,second.push(&pcm[..5600],false)?);
        assert_eq!(first.flush()?,second.flush()?);
        assert!(model.start_stream(Some(123),None,false).is_err());
        let mut invalid = model.start_stream(None,None,false)?;
        assert!(invalid.push(&[f32::NAN],false).is_err());
        assert!(invalid.flush().is_err());
    }
    Ok(())
}
