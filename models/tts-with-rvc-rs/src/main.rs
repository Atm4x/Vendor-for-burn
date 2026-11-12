use burn::tensor::Device;
use std::{env, fs, process::ExitCode};
mod user_cli;
use user_cli::{WaveJob, process_wave_job};
use tts_with_rvc_rs::{TextFrontFixture, attention::AttentionFixture, encoder::EncoderLayerFixture, text_encoder::TextEncoderFixture, max_abs_error};

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or("missing command")?;
    if matches!(command.as_str(), "convert" | "convert-batch" | "doctor" | "help" | "--help" | "-h") {
        return user_cli::run_named(&command, args);
    }
    if command == "inspect-pth" {
        let path = args.next().ok_or("missing checkpoint path")?;
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open(&path)?;
        println!("checkpoint={} tensor_count={}", path, checkpoint.tensor_count());
        for name in args {
            let (shape, values) = checkpoint.tensor_f32(&name)?;
            println!("tensor={name} shape={shape:?} elements={} first={:?}", values.len(), &values[..values.len().min(4)]);
        }
        return Ok(());
    }
    if command == "inspect-assets" {
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, faiss_index::IvfFlatIndex};
        let voice_path = args.next().ok_or("missing voice .pth")?;
        let hubert_path = args.next().ok_or("missing HuBERT .pt")?;
        let rmvpe_path = args.next().ok_or("missing RMVPE .pt")?;
        let index_path = args.next().unwrap_or_else(|| "-".to_owned());
        if args.next().is_some() {
            return Err("inspect-assets accepts voice.pth hubert.pt rmvpe.pt [index|-]".into());
        }
        let voice = RvcCheckpoint::open(&voice_path)?;
        let (speaker_shape, _) = voice.tensor_f32("emb_g.weight")?;
        let phone = voice.tensor_f32("enc_p.emb_phone.weight")?.0;
        if speaker_shape.len() != 2 || phone.len() != 2 || phone[1] != 768 {
            return Err("unsupported RVC voice, expected v2 768-channel content".into());
        }
        let hubert = RvcCheckpoint::open_model(&hubert_path)?;
        let hubert_first = hubert.tensor_f32("feature_extractor.conv_layers.0.0.weight")?.0;
        let rmvpe = RvcCheckpoint::open_root(&rmvpe_path)?;
        let rmvpe_first = rmvpe.tensor_f32("unet.encoder.layers.0.conv.0.conv.0.weight")?.0;
        let index = if index_path == "-" { None } else { Some(IvfFlatIndex::open(index_path)?) };
        if let Some(index) = &index {
            if index.dimension() != 768 {
                return Err(format!("RVC v2 expects 768-dim index, found {}", index.dimension()));
            }
        }
        println!("{}", serde_json::json!({
            "voice_tensors": voice.tensor_count(),
            "voice_speakers": speaker_shape[0],
            "content_channels": phone[1],
            "hubert_tensors": hubert.tensor_count(),
            "hubert_first_conv_shape": hubert_first,
            "rmvpe_tensors": rmvpe.tensor_count(),
            "rmvpe_first_conv_shape": rmvpe_first,
            "faiss": index.map(|index| serde_json::json!({
                "dimension": index.dimension(), "vectors": index.len(),
                "nlist": index.nlist(), "nprobe": index.nprobe(),
            })),
        }));
        return Ok(());
    }
    if command == "infer-wav-segment" || command == "infer-wav-segment-benchmark" {
        use tts_with_rvc_rs::{audio_io, rvc_inference::{InferenceOptions, RvcV2Inferencer}};
        let voice = args.next().ok_or("missing RVC .pth")?;
        let hubert = args.next().ok_or("missing HuBERT .pt")?;
        let rmvpe = args.next().ok_or("missing RMVPE .pt")?;
        let index = args.next().ok_or("missing RVC .index (use - to disable)")?;
        let source = args.next().ok_or("missing input WAV")?;
        let target = args.next().ok_or("missing output WAV")?;
        let start: f64 = args.next().ok_or("missing segment start seconds")?.parse().map_err(|e| format!("invalid start: {e}"))?;
        let duration: f64 = args.next().ok_or("missing segment duration seconds")?.parse().map_err(|e| format!("invalid duration: {e}"))?;
        if !start.is_finite() || !duration.is_finite() || start < 0.0 || duration < 0.2 || duration > 9.0 {
            return Err("start and duration must be finite, duration in [0.2, 9.0] seconds".into());
        }
        let wave = audio_io::read_mono_wav(&source)?;
        let begin = (start * wave.sample_rate as f64) as usize;
        let end = (((start + duration) * wave.sample_rate as f64) as usize).min(wave.samples.len());
        if begin >= end { return Err("specified WAV segment is outside the file".into()); }
        let input = audio_io::resample(&wave.samples[begin..end], wave.sample_rate, 16_000)?;
        let mut options = InferenceOptions::default();
        if index == "-" { options.index_rate = 0.0; }
        if let Some(rate) = args.next() {
            options.index_rate = rate.parse().map_err(|e| format!("invalid index rate: {e}"))?;
        }
        if let Some(shift) = args.next() {
            options.pitch_shift_semitones = shift.parse().map_err(|e| format!("invalid pitch shift: {e}"))?;
        }
        let backend = args.next().unwrap_or_else(|| "cpu".to_owned());
        let runs = if command == "infer-wav-segment-benchmark" {
            args.next().unwrap_or_else(|| "8".to_owned()).parse::<usize>()
                .map_err(|e| format!("invalid benchmark iterations: {e}"))?
        } else {
            1
        };
        if !(1..=64).contains(&runs) || args.next().is_some() {
            return Err("segment benchmark iterations must be 1..64 and no extra arguments are allowed".into());
        }
        let device = user_cli::selected_device(&backend)?;
        let load_start = std::time::Instant::now();
        let inferencer = RvcV2Inferencer::load(&voice, &hubert, &rmvpe,
            (index != "-").then_some(index.as_str()), device.clone())?;
        device.sync().map_err(|e| format!("model load synchronization failed: {e}"))?;
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
        let prepare_start = std::time::Instant::now();
        let filtered = tts_with_rvc_rs::preprocess::highpass_48hz_zero_phase(&input)?;
        let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;
        let mut timings = Vec::with_capacity(runs);
        let mut converted = Vec::new();
        for iteration in 0..runs {
            device.sync().map_err(|e| format!("pre-inference synchronization failed: {e}"))?;
            let start = std::time::Instant::now();
            converted = inferencer.infer_16khz(&filtered, &options)?;
            device.sync().map_err(|e| format!("post-inference synchronization failed: {e}"))?;
            let inference_ms = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("profile resident_inference iteration={iteration} inference_ms={inference_ms:.6} samples={}", converted.len());
            timings.push(inference_ms);
        }
        audio_io::write_mono_wav(&target, &converted, 40_000)?;
        if command == "infer-wav-segment-benchmark" {
            let mut warmed = timings.iter().skip(1).copied().collect::<Vec<_>>();
            warmed.sort_by(f64::total_cmp);
            let median = if warmed.is_empty() { None } else {
                let mid = warmed.len() / 2;
                Some(if warmed.len() % 2 == 0 { (warmed[mid - 1] + warmed[mid]) / 2.0 } else { warmed[mid] })
            };
            let report = serde_json::json!({
                "stage": 62,
                "backend": backend,
                "load_ms": load_ms,
                "prepare_ms": prepare_ms,
                "inference_ms": timings,
                "warm_median_ms": median,
                "warm_definition": "iterations 1..N in same process and same loaded model; synchronized full-output readback",
                "gpu_faiss": std::env::var_os("RVC_GPU_FAISS").is_some(),
                "gpu_f0": std::env::var_os("RVC_GPU_F0").is_some(),
                "graph_replay": false,
                "output_samples": converted.len(),
            });
            let report_path = format!("{target}.benchmark.json");
            fs::write(&report_path, serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
                .map_err(|e| format!("cannot save benchmark report: {e}"))?;
            println!("RVC resident benchmark: {report_path}");
        }
        println!("RVC WAV→WAV complete: samples={} sr=40000 path={target}", converted.len());
        return Ok(());
    }
    if command == "infer-wav-full-diagnostic" {
        use tts_with_rvc_rs::{
            audio_io, chunked::ChunkConfig, diagnostic,
            rvc_inference::{InferenceOptions, RvcV2Inferencer},
        };
        let voice = args.next().ok_or("missing RVC .pth")?;
        let hubert = args.next().ok_or("missing HuBERT .pt")?;
        let rmvpe = args.next().ok_or("missing RMVPE .pt")?;
        let index = args.next().ok_or("missing RVC .index")?;
        let source = args.next().ok_or("missing input WAV")?;
        let target = args.next().ok_or("missing output WAV")?;
        let backend = args.next().ok_or("missing backend: cuda")?;
        let core_frames = args.next().ok_or("missing core frames")?.parse::<usize>()
            .map_err(|e| format!("invalid core frames: {e}"))?;
        let batch_size = args.next().ok_or("missing tensor batch size")?.parse::<usize>()
            .map_err(|e| format!("invalid tensor batch size: {e}"))?;
        if args.next().is_some() || backend != "cuda" || index == "-"
            || source == target || !(1..=32).contains(&batch_size)
        {
            return Err("Stage64 accepts only cuda, a real index, one full WAV and batch size 1..32".into());
        }
        for incompatible in [
            "RVC_GPU_FAISS", "RVC_GPU_F0", "RVC_RMVPE_COMPACT_READBACK",
            "RVC_RMVPE_FUSED_READBACK", "RVC_RMVPE_PLANE_READBACK",
        ] {
            if env::var_os(incompatible).is_some() {
                return Err(format!("Stage64 rejects experimental mode {incompatible}"));
            }
        }
        if !diagnostic::enabled() {
            return Err("Stage64 requires RVC_DIAGNOSTIC_TRACE=1; detailed stage timing is mandatory".into());
        }
        let config = ChunkConfig { core_frames, ..ChunkConfig::default() };
        config.validate()?;
        let device = user_cli::selected_device(&backend)?;
        let opts = InferenceOptions::default();
        diagnostic::emit("benchmark_config", serde_json::json!({
            "backend":backend,"index_rate":opts.index_rate,"protect":opts.protect,
            "core_frames":core_frames,"tensor_batch_size":batch_size,
            "graph_replay":false,"gpu_ivf":false,"gpu_f0":false,
            "single_mode":true,"measured_full_passes":1,
        }));
        let started = std::time::Instant::now();
        let inferencer = RvcV2Inferencer::load(&voice, &hubert, &rmvpe, Some(index.as_str()), device.clone())?;
        device.sync().map_err(|e| format!("model load sync: {e}"))?;
        let load_ms = diagnostic::ms(started);
        diagnostic::emit("model_load_complete", serde_json::json!({"load_ms":load_ms}));

        let started = std::time::Instant::now();
        let wave = audio_io::read_mono_wav(&source)?;
        let original_seconds = wave.samples.len() as f64 / wave.sample_rate as f64;
        let resampled = audio_io::resample(&wave.samples, wave.sample_rate, 16000)?;
        let filtered = tts_with_rvc_rs::preprocess::highpass_48hz_zero_phase(&resampled)?;
        let prepare_ms = diagnostic::ms(started);
        let model_seconds = filtered.len() as f64 / 16000.0;
        if filtered.len() < 3200 { return Err("full diagnostic requires 0.2 seconds of input".into()); }
        diagnostic::emit("input_prepared", serde_json::json!({
            "source_seconds":original_seconds, "model_seconds":model_seconds,
            "source_hz":wave.sample_rate,"source_samples":wave.samples.len(),
            "model_input_samples":filtered.len(),"prepare_ms":prepare_ms,
        }));

        diagnostic::set_phase("warmup");
        let warmup_samples = (core_frames + 2 * config.context_frames + config.trailing_frames)
            .checked_mul(160).ok_or("warmup size overflow")?;
        let warmup = (0..warmup_samples).map(|i| filtered[i % filtered.len()]).collect::<Vec<_>>();
        let warm_inputs = vec![warmup; batch_size];
        device.sync().map_err(|e| format!("pre-warmup sync: {e}"))?;
        let started = std::time::Instant::now();
        let warm_options = vec![opts.clone(); batch_size];
        let warm_output = inferencer.infer_16khz_batch(&warm_inputs, &warm_options)?;
        device.sync().map_err(|e| format!("warmup sync: {e}"))?;
        let warmup_ms = diagnostic::ms(started);
        diagnostic::emit("representative_shape_warmup", serde_json::json!({
            "warmup_ms":warmup_ms,"warmup_batch":batch_size,
            "warmup_samples_per_item":warmup_samples,
            "warmup_outputs":warm_output.len(),
            "note":"one representative batch excluded from the single full-WAV benchmark",
        }));

        device.sync().map_err(|e| format!("pre-inference sync: {e}"))?;
        diagnostic::set_phase("measured");
        diagnostic::emit("measured_full_pass_begin", serde_json::json!({"source_seconds":model_seconds}));
        let started = std::time::Instant::now();
        let mut chunk_count = 0usize;
        let mut converted = inferencer.infer_long_filtered_16khz_batched(
            &filtered, &opts, config, batch_size, |progress| {
                chunk_count = progress.chunk_count;
                diagnostic::emit("chunk_complete", serde_json::json!({
                    "chunk_index":progress.chunk_index,
                    "chunk_count":progress.chunk_count,
                    "output_samples":progress.output_samples,
                }));
            },
        )?;
        device.sync().map_err(|e| format!("post-inference sync: {e}"))?;
        let inference_ms = diagnostic::ms(started);
        let expected_samples = filtered.len().checked_mul(5).ok_or("output length overflow")? / 2;
        if converted.len() != expected_samples || converted.iter().any(|v| !v.is_finite()) {
            return Err(format!("full-WAV output invalid: got {}, expected {expected_samples}", converted.len()));
        }
        diagnostic::emit("measured_full_pass_end", serde_json::json!({
            "inference_ms":inference_ms,"rtf":inference_ms / (model_seconds*1000.0),
            "chunks":chunk_count,"samples":converted.len(),
        }));
        diagnostic::set_phase("postprocess");
        let started = std::time::Instant::now();
        audio_io::mix_rms_envelope(&filtered, 16000, &mut converted, 40000, 0.5)?;
        audio_io::normalize_peak(&mut converted)?;
        audio_io::write_mono_wav(&target, &converted, 40000)?;
        let postprocess_write_ms = diagnostic::ms(started);
        diagnostic::emit("postprocess_write", serde_json::json!({
            "postprocess_write_ms":postprocess_write_ms,"output_samples":converted.len(),
        }));
        let report = serde_json::json!({
            "stage":64,"benchmark":"single_cuda_full_wav_diagnostic",
            "mode":"cpu_faiss_cpu_fp64_f0","backend":backend,
            "model_input_seconds":model_seconds,"source_seconds":original_seconds,
            "output_seconds":converted.len() as f64 / 40000.0,
            "source_sample_rate":wave.sample_rate,"output_sample_rate":40000,
            "output_samples":converted.len(),"chunk_count":chunk_count,
            "core_frames":config.core_frames,"overlap_frames":config.overlap_frames,
            "context_frames":config.context_frames,"tensor_batch_size":batch_size,
            "load_ms":load_ms,"prepare_ms":prepare_ms,"warmup_ms":warmup_ms,
            "full_inference_ms":inference_ms,"rtf":inference_ms / (model_seconds*1000.0),
            "postprocess_write_ms":postprocess_write_ms,"measured_full_passes":1,
            "graph_replay":false,"gpu_faiss":false,"gpu_f0":false,
            "profiling":"synchronization after each traced stage, including nested stages; measured timing includes instrumentation overhead and must not be compared directly against unprofiled Stage63 RTF",
        });
        let report_path = format!("{target}.diagnostic.json");
        fs::write(&report_path, serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
            .map_err(|e| format!("cannot write diagnostic report: {e}"))?;
        println!("Stage64 full CUDA diagnostic: {model_seconds:.3}s input, {inference_ms:.3}ms inference, RTF {:.4}; report={report_path}", inference_ms/(model_seconds*1000.0));
        return Ok(());
    }
    if command == "infer-wav-full-benchmark" {
        use tts_with_rvc_rs::{
            audio_io, chunked::ChunkConfig,
            rvc_inference::{InferenceOptions, RvcV2Inferencer},
        };
        let voice = args.next().ok_or("missing RVC .pth")?;
        let hubert = args.next().ok_or("missing HuBERT .pt")?;
        let rmvpe = args.next().ok_or("missing RMVPE .pt")?;
        let index = args.next().ok_or("missing RVC .index (use - to disable)")?;
        let source = args.next().ok_or("missing full input WAV")?;
        let target = args.next().ok_or("missing output WAV")?;
        if source == target {
            return Err("benchmark must not overwrite the input WAV".into());
        }
        let index_rate: f32 = args.next().ok_or("missing index rate")?
            .parse().map_err(|e| format!("invalid index rate: {e}"))?;
        let backend = args.next().ok_or("missing backend")?;
        let runs: usize = args.next().ok_or("missing full inference runs")?
            .parse().map_err(|e| format!("invalid runs: {e}"))?;
        let core_frames: usize = args.next().ok_or("missing core frames")?
            .parse().map_err(|e| format!("invalid core frames: {e}"))?;
        let tensor_batch_size: usize = args.next().ok_or("missing tensor batch size")?
            .parse().map_err(|e| format!("invalid tensor batch size: {e}"))?;
        if args.next().is_some() || !(2..=16).contains(&runs) || !(1..=32).contains(&tensor_batch_size) {
            return Err("full benchmark requires 2..16 runs and 1..32 tensor batch size, without extra arguments".into());
        }
        if !index_rate.is_finite() || !(0.0..=1.0).contains(&index_rate)
            || (index_rate > 0.0 && index == "-") {
            return Err("index rate requires a real index and must be in 0..1".into());
        }
        let config = ChunkConfig { core_frames, ..ChunkConfig::default() };
        config.validate()?;
        let device = user_cli::selected_device(&backend)?;
        let mut opts = InferenceOptions::default();
        opts.index_rate = index_rate;

        let load_start = std::time::Instant::now();
        let inferencer = RvcV2Inferencer::load(
            &voice, &hubert, &rmvpe, (index != "-").then_some(index.as_str()), device.clone()
        )?;
        device.sync().map_err(|e| format!("model load sync failed: {e}"))?;
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

        let prepare_start = std::time::Instant::now();
        let wave = audio_io::read_mono_wav(&source)?;
        let pcm = audio_io::resample(&wave.samples, wave.sample_rate, 16_000)?;
        let filtered = tts_with_rvc_rs::preprocess::highpass_48hz_zero_phase(&pcm)?;
        let source_seconds = wave.samples.len() as f64 / wave.sample_rate as f64;
        let model_seconds = filtered.len() as f64 / 16_000.0;
        if filtered.len() < 3_200 {
            return Err("full benchmark requires at least 0.2 seconds of source audio".into());
        }
        let expected_samples = filtered.len().checked_mul(5)
            .ok_or("full benchmark output size overflow")? / 2;
        let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;
        let mut timings = Vec::with_capacity(runs);
        let mut previous: Option<Vec<f32>> = None;
        let mut max_iteration_delta = 0.0_f32;
        let mut exact_iterations = true;
        let mut chunk_counts = Vec::with_capacity(runs);
        let mut converted = Vec::new();
        for iteration in 0..runs {
            device.sync().map_err(|e| format!("pre-inference sync failed: {e}"))?;
            let start = std::time::Instant::now();
            let mut count = 0_usize;
            converted = inferencer.infer_long_filtered_16khz_batched(
                &filtered, &opts, config, tensor_batch_size,
                |progress| {
                    count = progress.chunk_count;
                    eprintln!("profile full_wav iteration={iteration} chunk={}/{} output_samples={}",
                        progress.chunk_index, progress.chunk_count, progress.output_samples);
                },
            )?;
            device.sync().map_err(|e| format!("post-inference sync failed: {e}"))?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            if converted.len() != expected_samples || converted.iter().any(|x| !x.is_finite()) {
                return Err(format!("full RVC output is invalid: got {} samples, expected {expected_samples}", converted.len()));
            }
            if let Some(old) = &previous {
                for (a,b) in old.iter().zip(&converted) {
                    if a.to_bits() != b.to_bits() { exact_iterations = false; }
                    max_iteration_delta = max_iteration_delta.max((a-b).abs());
                }
            }
            previous = Some(converted.clone());
            timings.push(elapsed_ms);
            chunk_counts.push(count);
            eprintln!("profile full_wav iteration={iteration} elapsed_ms={elapsed_ms:.3} source_seconds={model_seconds:.3} chunks={count}");
        }
        let write_start = std::time::Instant::now();
        audio_io::mix_rms_envelope(&filtered, 16_000, &mut converted, 40_000, 0.5)?;
        audio_io::normalize_peak(&mut converted)?;
        audio_io::write_mono_wav(&target, &converted, 40_000)?;
        let postprocess_write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
        let mut warmed = timings.iter().skip(1).copied().collect::<Vec<_>>();
        warmed.sort_by(f64::total_cmp);
        let mid = warmed.len()/2;
        let warm_median_ms = if warmed.len()%2 == 0 { (warmed[mid-1]+warmed[mid])/2.0 } else { warmed[mid] };
        let report = serde_json::json!({
            "stage": 63,
            "benchmark": "full-wav-chunked-resident-rvc",
            "backend": backend,
            "graph_replay": false,
            "source_seconds": source_seconds,
            "model_input_seconds": model_seconds,
            "source_samples": wave.samples.len(),
            "source_sample_rate": wave.sample_rate,
            "output_samples": converted.len(),
            "output_sample_rate": 40000,
            "load_ms": load_ms,
            "prepare_ms": prepare_ms,
            "postprocess_write_ms": postprocess_write_ms,
            "inference_ms": timings,
            "first_inference_ms": timings[0],
            "warm_median_ms": warm_median_ms,
            "warm_rtf": warm_median_ms / (model_seconds * 1000.0),
            "iterations_exact_f32": exact_iterations,
            "max_abs_iteration_delta": max_iteration_delta,
            "chunk_counts": chunk_counts,
            "core_frames": config.core_frames,
            "overlap_frames": config.overlap_frames,
            "context_frames": config.context_frames,
            "tensor_batch_size": tensor_batch_size,
            "gpu_faiss": std::env::var_os("RVC_GPU_FAISS").is_some(),
            "gpu_f0": std::env::var_os("RVC_GPU_F0").is_some(),
            "global_f0": std::env::var_os("RVC_GLOBAL_F0").is_some_and(|value| value == "1"),
            "rmvpe_gru_execution": if std::env::var("RVC_RMVPE_GRU_PERSISTENT").as_deref() == Ok("1") {"persistent"} else {"stepwise"},
            "nsf_conv_execution": if std::env::var("RVC_NSF_IMPLICIT_GEMM").as_deref() == Ok("1")
                || (std::env::var("RVC_NSF_CONV1D_GEMM").as_deref() == Ok("1")
                    && std::env::var("RVC_NSF_TRANSPOSE_GEMM").as_deref() == Ok("1"))
                {"implicit_gemm"} else if std::env::var("RVC_NSF_CONV1D_GEMM").as_deref() == Ok("1")
                    || std::env::var("RVC_NSF_TRANSPOSE_GEMM").as_deref() == Ok("1")
                {"mixed"} else {"direct"},
            "nsf_conv1d_execution": if std::env::var("RVC_NSF_IMPLICIT_GEMM").as_deref() == Ok("1")
                || std::env::var("RVC_NSF_CONV1D_GEMM").as_deref() == Ok("1")
                {"implicit_gemm"} else {"direct"},
            "nsf_transpose_execution": if std::env::var("RVC_NSF_IMPLICIT_GEMM").as_deref() == Ok("1")
                || std::env::var("RVC_NSF_TRANSPOSE_GEMM").as_deref() == Ok("1")
                {"gemm"} else {"direct"},
            "hubert_frontend_conv_execution": if std::env::var("RVC_HUBERT_FRONTEND_GEMM").as_deref() == Ok("1") {"implicit_gemm"} else {"direct"},
            "rmvpe_unet_conv_execution": if std::env::var("RVC_RMVPE_UNET_GEMM").as_deref() == Ok("1") {"implicit_gemm"} else if std::env::var("RVC_RMVPE_UNET_CONV2D_GEMM").as_deref() == Ok("1") && std::env::var("RVC_RMVPE_UNET_TRANSPOSE2D_GEMM").as_deref() == Ok("1") {"implicit_gemm"} else if std::env::var("RVC_RMVPE_UNET_CONV2D_GEMM").as_deref() == Ok("1") {"conv2d_implicit_gemm"} else if std::env::var("RVC_RMVPE_UNET_TRANSPOSE2D_GEMM").as_deref() == Ok("1") {"transpose2d_gemm"} else {"direct"},
            "nsf_temporal_conv_kernel": std::env::var("RVC_NSF_CONV_KERNEL").unwrap_or_default(),
            "nsf_execution": "sequential",
            "warm_definition": "same process, identical full WAV, persistent weights, synchronized outputs; iterations 1..N; no CUDA Graph",
        });
        let path = format!("{target}.benchmark.json");
        fs::write(&path, serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
            .map_err(|e| format!("cannot save full WAV benchmark report: {e}"))?;
        println!("Full RVC completed, {:.3}s source => {:.3}s output; warm median {:.3}ms, RTF {:.4}; report={path}",
            source_seconds, converted.len() as f64 / 40000.0, warm_median_ms, warm_median_ms/(model_seconds*1000.0));
        return Ok(());
    }
    if command == "infer-wav-full" || command == "infer-wav-batch" {
        use tts_with_rvc_rs::{
            chunked::ChunkConfig,
            rvc_inference::{InferenceOptions, RvcV2Inferencer},
        };
        let voice = args.next().ok_or("missing RVC .pth")?;
        let hubert = args.next().ok_or("missing HuBERT .pt")?;
        let rmvpe = args.next().ok_or("missing RMVPE .pt")?;
        let index = args.next().ok_or("missing RVC .index (use - to disable)")?;
        let jobs = if command == "infer-wav-full" {
            vec![WaveJob {
                input: args.next().ok_or("missing input WAV")?,
                output: args.next().ok_or("missing output WAV")?,
                index_rate: None,
                pitch_shift_semitones: None,
                rms_mix_rate: None,
            }]
        } else {
            let manifest = args.next().ok_or("missing batch JSON manifest")?;
            let jobs: Vec<WaveJob> =
                serde_json::from_slice(&fs::read(manifest).map_err(|e| e.to_string())?)
                    .map_err(|e| format!("invalid RVC batch manifest: {e}"))?;
            if jobs.is_empty() {
                return Err("RVC batch manifest must have at least one job".into());
            }
            jobs
        };
        let mut options = InferenceOptions::default();
        if index == "-" {
            options.index_rate = 0.0;
        }
        if let Some(v) = args.next() {
            options.index_rate = v.parse().map_err(|e| format!("invalid index rate: {e}"))?;
        }
        if let Some(v) = args.next() {
            options.pitch_shift_semitones =
                v.parse().map_err(|e| format!("invalid pitch shift: {e}"))?;
        }
        let mut config = ChunkConfig::default();
        if let Some(v) = args.next() {
            config.core_frames = v.parse().map_err(|e| format!("invalid core frames: {e}"))?;
        }
        if let Some(v) = args.next() {
            config.overlap_frames = v
                .parse()
                .map_err(|e| format!("invalid overlap frames: {e}"))?;
        }
        if let Some(v) = args.next() {
            config.context_frames = v
                .parse()
                .map_err(|e| format!("invalid context frames: {e}"))?;
        }
        let tensor_batch_size: usize = args.next().map(|value| value.parse().map_err(|e| format!("invalid tensor batch size: {e}")))
            .transpose()?.unwrap_or(2);
        if !(1..=32).contains(&tensor_batch_size) {
            return Err("tensor batch size must be 1..=32".into());
        }
        config.validate()?;
        let mut outputs = std::collections::HashSet::new();
        let inputs: std::collections::HashSet<_> =
            jobs.iter().map(|job| job.input.as_str()).collect();
        for job in &jobs {
            if job.input.is_empty()
                || job.output.is_empty()
                || inputs.contains(job.output.as_str())
                || !outputs.insert(job.output.as_str())
            {
                return Err(
                    "RVC batch contains an empty/duplicate output or overwrites an input".into(),
                );
            }
        }
        let backend = args.next().unwrap_or_else(|| "cpu".to_owned());
        let device = user_cli::selected_device(&backend)?;
        let inferencer = RvcV2Inferencer::load(
            &voice,
            &hubert,
            &rmvpe,
            (index != "-").then_some(index.as_str()),
            device,
        )?;
        for (i, job) in jobs.iter().enumerate() {
            eprintln!("RVC batch job {}/{}", i + 1, jobs.len());
            process_wave_job(&inferencer, job, &options, config, tensor_batch_size)?;
        }
        return Ok(());
    }
    if command == "inspect-index" {
        let path = args.next().ok_or("missing FAISS .index path")?;
        let index = tts_with_rvc_rs::faiss_index::IvfFlatIndex::open(&path)?;
        println!("faiss=IndexIVFFlat metric=L2 dimension={} vectors={} nlist={} nprobe={}",
            index.dimension(), index.len(), index.nlist(), index.nprobe());
        if let Some(sample) = index.reconstruct(0) {
            let result = index.search(sample, 8)?;
            println!("probe_id=0 nearest={:?}", result);
        }
        return Ok(());
    }
    if !matches!(command.as_str(), "verify-text-front" | "verify-attention" | "verify-encoder-layer" | "verify-text-encoder" | "verify-flow" | "verify-text-encoder-pth" | "verify-decoder-stage" | "verify-nsf-decoder" | "verify-nsf-source" | "verify-nsf-from-f0" | "verify-hubert-conv" | "verify-hubert-frontend" | "verify-hubert-transformer" | "verify-hubert-stack" | "verify-hubert-position" | "verify-hubert-encoder" | "verify-hubert-model" | "verify-rmvpe-head" | "verify-rmvpe-encoder" | "verify-rmvpe-intermediate" | "verify-rmvpe-decoder" | "verify-rmvpe-unet" | "verify-rmvpe-gru" | "verify-rmvpe-network" | "verify-rmvpe-mel" | "verify-rmvpe-audio") {
        return Err("usage: tts-with-rvc-rs verify-text-front|verify-attention|verify-encoder-layer|verify-text-encoder <fixture.json> [flex|cpu|cuda|wgpu]".into());
    }
    let path = args.next().ok_or("missing fixture or checkpoint path")?;
    let flow_fixture_path = if command == "verify-flow" || command == "verify-text-encoder-pth" || command == "verify-decoder-stage" || command == "verify-nsf-decoder" || command == "verify-nsf-source" || command == "verify-nsf-from-f0" || command == "verify-hubert-conv" || command == "verify-hubert-frontend" || command == "verify-hubert-transformer" || command == "verify-hubert-stack" || command == "verify-hubert-position" || command == "verify-hubert-encoder" || command == "verify-hubert-model" || command == "verify-rmvpe-head" || command == "verify-rmvpe-encoder" || command == "verify-rmvpe-intermediate" || command == "verify-rmvpe-decoder" || command == "verify-rmvpe-unet" || command == "verify-rmvpe-gru" || command == "verify-rmvpe-network" || command == "verify-rmvpe-audio" {
        Some(args.next().ok_or("missing flow fixture path")?)
    } else { None };
    let backend = args.next().unwrap_or_else(|| "cpu".to_owned());
    let device = match backend.as_str() {
        #[cfg(feature = "flex")]
        "flex" => Device::flex(),
        #[cfg(feature = "cpu")]
        "cpu" => Device::cpu(),
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(Default::default()),
        other => return Err(format!("backend {other} is not compiled into this binary")),
    };
    if command == "verify-hubert-conv" {
        use burn::tensor::{Tensor, TensorData};
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open_model(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing HuBERT reference")?;
        let fixture: serde_json::Value = serde_json::from_str(&fs::read_to_string(fixture_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["waveform"].clone()).map_err(|e| e.to_string())?;
        let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).map_err(|e| e.to_string())?;
        let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).map_err(|e| e.to_string())?;
        let model = tts_with_rvc_rs::hubert::HubertFirstConv::load(&checkpoint, &device)?;
        let actual = model.forward(Tensor::from_data(TensorData::new(input, shape), &device))
            .into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &expected);
        println!("backend={backend} max_abs_hubert_conv={error:.9} tensor_count={}", checkpoint.tensor_count());
        if !error.is_finite() || error > 1e-4 { return Err("HuBERT Conv1d/GroupNorm/GELU parity exceeded tolerance".into()); }
        return Ok(());
    }
    if command == "verify-hubert-frontend" {
        use burn::tensor::{Tensor, TensorData};
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open_model(&path)?;
        let reference_path = flow_fixture_path.as_ref().ok_or("missing frontend reference")?;
        let fixture: serde_json::Value = serde_json::from_str(&fs::read_to_string(reference_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let waveform: Vec<f32> = serde_json::from_value(fixture["waveform"].clone()).map_err(|e| e.to_string())?;
        let input_shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).map_err(|e| e.to_string())?;
        let expected_shape: [usize; 3] = serde_json::from_value(fixture["expected_shape"].clone()).map_err(|e| e.to_string())?;
        let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).map_err(|e| e.to_string())?;
        let model = tts_with_rvc_rs::hubert::HubertFrontend::load(&checkpoint, &device)?;
        let result = model.forward(Tensor::from_data(TensorData::new(waveform, input_shape), &device));
        if result.dims() != expected_shape {
            return Err(format!("HuBERT frontend expected shape {expected_shape:?}, got {:?}", result.dims()));
        }
        let actual = result.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &expected);
        println!("backend={backend} max_abs_hubert_frontend={error:.9} tensor_count={}", checkpoint.tensor_count());
        if !error.is_finite() || error > 1e-4 {
            return Err("HuBERT frontend Torch/Burn parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-hubert-transformer" || command == "verify-hubert-stack" {
        use burn::tensor::{Tensor, TensorData};
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open_model(&path)?;
        let reference_path = flow_fixture_path.as_ref().ok_or("missing transformer reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(reference_path).map_err(|e| e.to_string())?,
        ).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).map_err(|e| e.to_string())?;
        let valid: Vec<f32> = serde_json::from_value(fixture["valid"].clone()).map_err(|e| e.to_string())?;
        let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).map_err(|e| e.to_string())?;
        let unmasked: Vec<f32> = serde_json::from_value(fixture["expected_unmasked"].clone()).map_err(|e| e.to_string())?;
        let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).map_err(|e| e.to_string())?;
        let heads: usize = serde_json::from_value(fixture["heads"].clone()).map_err(|e| e.to_string())?;
        let input = Tensor::from_data(TensorData::new(input, shape), &device);
        let valid = Tensor::from_data(TensorData::new(valid, [shape[0], shape[1]]), &device);
        let (actual, actual_unmasked) = if command == "verify-hubert-stack" {
            let layers: usize = serde_json::from_value(fixture["layers"].clone()).map_err(|e| e.to_string())?;
            let stack = tts_with_rvc_rs::hubert_transformer::HubertTransformerStack::load(&checkpoint, layers, heads, &device)?;
            (stack.forward(input.clone(), Some(valid)), stack.forward(input, None))
        } else {
            let layer = tts_with_rvc_rs::hubert_transformer::HubertTransformerLayer::load(&checkpoint, 0, heads, &device)?;
            (layer.forward(input.clone(), Some(valid)), layer.forward(input, None))
        };
        let actual = actual.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let actual_unmasked = actual_unmasked.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &expected);
        let unmasked_error = max_abs_error(&actual_unmasked, &unmasked);
        println!("backend={backend} max_abs_{command}={error:.9} max_abs_unmasked={unmasked_error:.9}");
        if !error.is_finite() || error > 1e-4 || !unmasked_error.is_finite() || unmasked_error > 1e-4 {
            return Err("HuBERT Transformer Torch/Burn parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-hubert-model" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, hubert_model::HubertFeatureModel};
        let checkpoint = RvcCheckpoint::open_model(&path)?;
        let reference_path = flow_fixture_path.as_ref().ok_or("missing HuBERT waveform reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(reference_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let read = |field: &str| -> Result<Vec<f32>, String> {
            serde_json::from_value(fixture[field].clone()).map_err(|error| error.to_string())
        };
        let input_shape: [usize; 3] = serde_json::from_value(fixture["input_shape"].clone()).map_err(|error| error.to_string())?;
        let output_shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).map_err(|error| error.to_string())?;
        let heads: usize = serde_json::from_value(fixture["heads"].clone()).map_err(|error| error.to_string())?;
        let layers: usize = serde_json::from_value(fixture["layers"].clone()).map_err(|error| error.to_string())?;
        let waveform = Tensor::from_data(TensorData::new(read("waveform")?, input_shape), &device);
        let valid = Tensor::from_data(TensorData::new(read("valid")?, [output_shape[0], output_shape[1]]), &device);
        let model = HubertFeatureModel::load(&checkpoint, layers, heads, &device)?;
        let result = model.forward(waveform.clone(), Some(valid));
        let unmasked_result = model.forward(waveform, None);
        if result.dims() != output_shape || unmasked_result.dims() != output_shape {
            return Err("HuBERT complete waveform forward produced incorrect output shape".into());
        }
        let actual = result.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let unmasked = unmasked_result.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let error = max_abs_error(&actual, &read("expected")?);
        let unmasked_error = max_abs_error(&unmasked, &read("expected_unmasked")?);
        println!("backend={backend} max_abs_hubert_model={error:.9} max_abs_unmasked={unmasked_error:.9} layers={layers} frames={}", output_shape[1]);
        if !error.is_finite() || !unmasked_error.is_finite() || error > 1e-4 || unmasked_error > 1e-4 {
            return Err(format!("HuBERT waveform → features parity exceeded tolerance: {error:.8}, {unmasked_error:.8}"));
        }
        return Ok(());
    }
    if command == "verify-hubert-position" || command == "verify-hubert-encoder" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, hubert_position::HubertEncoder};
        let checkpoint = RvcCheckpoint::open_model(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing positional reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(fixture_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let read = |field: &str| -> Result<Vec<f32>, String> {
            serde_json::from_value(fixture[field].clone()).map_err(|error| error.to_string())
        };
        let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).map_err(|error| error.to_string())?;
        let heads: usize = serde_json::from_value(fixture["heads"].clone()).map_err(|error| error.to_string())?;
        let layers: usize = serde_json::from_value(fixture["layers"].clone()).map_err(|error| error.to_string())?;
        let x = Tensor::from_data(TensorData::new(read("input")?, shape), &device);
        let valid = Tensor::from_data(TensorData::new(read("valid")?, [shape[0], shape[1]]), &device);
        let model = HubertEncoder::load(&checkpoint, layers, heads, &device)?;
        let (actual, unmasked, expected, expected_unmasked) = if command == "verify-hubert-position" {
            (model.position_only(x.clone(), Some(valid)), model.position_only(x, None),
             read("expected_position")?, read("expected_position_unmasked")?)
        } else {
            (model.forward(x.clone(), Some(valid)), model.forward(x, None),
             read("expected")?, read("expected_unmasked")?)
        };
        let actual = actual.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let unmasked = unmasked.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let error = max_abs_error(&actual, &expected);
        let unmasked_error = max_abs_error(&unmasked, &expected_unmasked);
        println!("backend={backend} max_abs_{command}={error:.9} max_abs_unmasked={unmasked_error:.9} layers={layers}");
        if !error.is_finite() || !unmasked_error.is_finite() || error > 1e-4 || unmasked_error > 1e-4 {
            return Err(format!("HuBERT positional/encoder parity exceeded tolerance: {error:.8}, {unmasked_error:.8}"));
        }
        return Ok(());
    }
    if command == "verify-rmvpe-mel" || command == "verify-rmvpe-audio" {
        use tts_with_rvc_rs::rmvpe_mel::{RmvpeAudioModel, RmvpeMelSpectrogram};
        let reference_path = if command == "verify-rmvpe-audio" {
            flow_fixture_path.as_ref().ok_or("missing RMVPE audio reference")?
        } else { &path };
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(reference_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let audio: Vec<f32> = serde_json::from_value(fixture["audio"].clone()).map_err(|error| error.to_string())?;
        let shape: [usize; 3] = serde_json::from_value(fixture["mel_shape"].clone()).map_err(|error| error.to_string())?;
        let expected_mel: Vec<f32> = serde_json::from_value(fixture["expected_mel"].clone()).map_err(|error| error.to_string())?;
        let (actual_mel, frames) = RmvpeMelSpectrogram::new().forward(&audio)?;
        if [1, 128, frames] != shape { return Err("RMVPE STFT/Mel shape mismatch".into()); }
        let mel_error = max_abs_error(&actual_mel, &expected_mel);
        println!("backend={backend} rmvpe_mel_frames={frames} max_abs_rmvpe_mel={mel_error:.9}");
        if !mel_error.is_finite() || mel_error > 5e-4 { return Err("RMVPE STFT/Mel numerical parity exceeded tolerance".into()); }
        if command == "verify-rmvpe-audio" {
            let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open_root(&path)?;
            let network = RmvpeAudioModel::load(&checkpoint, &device)?;
            let (salience, count) = network.forward_pcm(&audio, &device)?;
            if count != frames { return Err("RMVPE full audio frame count mismatch".into()); }
            let expected_shape: [usize; 3] = serde_json::from_value(fixture["expected_salience_shape"].clone()).map_err(|error| error.to_string())?;
            if salience.dims() != expected_shape { return Err(format!("RMVPE audio salience shape mismatch: {:?}", salience.dims())); }
            let expected: Vec<f32> = serde_json::from_value(fixture["expected_salience"].clone()).map_err(|error| error.to_string())?;
            let values = salience.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
            let error = max_abs_error(&values, &expected);
            println!("backend={backend} max_abs_rmvpe_audio_salience={error:.9}");
            if !error.is_finite() || error > 2e-3 { return Err("RMVPE PCM-to-salience numerical parity exceeded tolerance".into()); }
        }
        return Ok(());
    }
    if command == "verify-rmvpe-gru" || command == "verify-rmvpe-network" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe::salience_to_f0, rmvpe::RmvpeHead, rmvpe_gru::{RmvpeBiGru, RmvpeNetwork}};
        let checkpoint = RvcCheckpoint::open_root(&path)?;
        let reference_path = flow_fixture_path.as_ref().ok_or("missing RMVPE GRU reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(reference_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let read = |name: &str| -> Result<Vec<f32>, String> {
            serde_json::from_value(fixture[name].clone()).map_err(|error| error.to_string())
        };
        let (actual, expected, expected_f0) = if command == "verify-rmvpe-gru" {
            let shape: [usize; 3] = serde_json::from_value(fixture["gru_input_shape"].clone()).map_err(|error| error.to_string())?;
            let expected_shape: [usize; 3] = serde_json::from_value(fixture["gru_output_shape"].clone()).map_err(|error| error.to_string())?;
            let gru = RmvpeBiGru::load(&checkpoint, &device)?;
            let input = Tensor::from_data(TensorData::new(read("gru_input")?, shape), &device);
            let encoded = gru.forward(input);
            if encoded.dims() != expected_shape { return Err("RMVPE GRU output shape mismatch".into()); }
            let encoded = encoded.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
            let gru_error = max_abs_error(&encoded, &read("expected_gru")?);
            if !gru_error.is_finite() || gru_error > 1e-4 { return Err(format!("RMVPE bidirectional GRU error {gru_error}")); }
            println!("backend={backend} rmvpe_gru_input={} hidden={} max_abs_gru={gru_error:.9}", gru.input_size(), gru.hidden_size());
            let head = RmvpeHead::load(&checkpoint, &device)?;
            let output = head.forward(Tensor::from_data(TensorData::new(encoded, expected_shape), &device));
            (output.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?, read("expected_salience")?, read("expected_f0")?)
        } else {
            let shape: [usize; 3] = serde_json::from_value(fixture["mel_shape"].clone()).map_err(|error| error.to_string())?;
            let levels: usize = serde_json::from_value(fixture["unet_levels"].clone()).map_err(|error| error.to_string())?;
            let middle: usize = serde_json::from_value(fixture["middle_levels"].clone()).map_err(|error| error.to_string())?;
            let model = RmvpeNetwork::load(&checkpoint, levels, middle, shape[1], &device)?;
            let input = Tensor::from_data(TensorData::new(read("mel")?, shape), &device);
            let output = model.forward(input);
            let expected_shape: [usize; 3] = serde_json::from_value(fixture["expected_full_shape"].clone()).map_err(|error| error.to_string())?;
            if output.dims() != expected_shape { return Err(format!("RMVPE network output shape mismatch {:?}", output.dims())); }
            (output.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?, read("expected_full")?, read("expected_full_f0")?)
        };
        let error = max_abs_error(&actual, &expected);
        let f0 = salience_to_f0(&actual, 0.03)?;
        let f0_error = max_abs_error(&f0, &expected_f0);
        println!("backend={backend} max_abs_{command}={error:.9} max_abs_f0={f0_error:.9}");
        if !error.is_finite() || error > 1e-4 || !f0_error.is_finite() || f0_error > 1e-3 {
            return Err(format!("RMVPE GRU/network parity exceeded tolerance: {error}, {f0_error}"));
        }
        return Ok(());
    }
    if command == "verify-rmvpe-unet" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_unet::RmvpeUnet};
        let checkpoint = RvcCheckpoint::open_root(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing RMVPE full U-Net reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(fixture_path).map_err(|e| e.to_string())?,
        ).map_err(|e| e.to_string())?;
        let shape: [usize; 4] = serde_json::from_value(fixture["input_shape"].clone()).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).map_err(|e| e.to_string())?;
        let output_shape: [usize; 4] = serde_json::from_value(fixture["output_shape"].clone()).map_err(|e| e.to_string())?;
        let feature_shape: [usize; 3] = serde_json::from_value(fixture["feature_shape"].clone()).map_err(|e| e.to_string())?;
        let expected: Vec<f32> = serde_json::from_value(fixture["expected"].clone()).map_err(|e| e.to_string())?;
        let expected_features: Vec<f32> = serde_json::from_value(fixture["expected_features"].clone()).map_err(|e| e.to_string())?;
        let levels: usize = serde_json::from_value(fixture["encoder_levels"].clone()).map_err(|e| e.to_string())?;
        let middle: usize = serde_json::from_value(fixture["intermediate_levels"].clone()).map_err(|e| e.to_string())?;
        let model = RmvpeUnet::load(&checkpoint, levels, middle, &device)?;
        let x = Tensor::from_data(TensorData::new(input, shape), &device);
        let output = model.forward(x.clone());
        if output.dims() != output_shape { return Err(format!("RMVPE U-Net output shape mismatch: {:?}", output.dims())); }
        let actual = output.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &expected);
        let features = model.forward_gru_features(x);
        if features.dims() != feature_shape { return Err(format!("RMVPE CNN-to-BiGRU feature shape mismatch: {:?}", features.dims())); }
        let actual_features = features.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let feature_error = max_abs_error(&actual_features, &expected_features);
        println!("backend={backend} rmvpe_unet_levels={} max_abs_rmvpe_unet={error:.9} max_abs_rmvpe_cnn_features={feature_error:.9}", model.encoder_levels());
        if !error.is_finite() || !feature_error.is_finite() || error > 1e-4 || feature_error > 1e-4 {
            return Err("RMVPE full U-Net/CNN Torch/Burn parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-rmvpe-decoder" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_unet::RmvpeDecoder};
        let checkpoint = RvcCheckpoint::open_root(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing RMVPE decoder reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(fixture_path).map_err(|e| e.to_string())?,
        ).map_err(|e| e.to_string())?;
        let input_shape: [usize; 4] = serde_json::from_value(fixture["input_shape"].clone()).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).map_err(|e| e.to_string())?;
        let mut skips = Vec::new();
        for item in fixture["skips"].as_array().ok_or("missing RMVPE decoder skips")? {
            let shape: [usize; 4] = serde_json::from_value(item["shape"].clone()).map_err(|e| e.to_string())?;
            let values: Vec<f32> = serde_json::from_value(item["values"].clone()).map_err(|e| e.to_string())?;
            skips.push(Tensor::from_data(TensorData::new(values, shape), &device));
        }
        let levels = fixture["levels"].as_array().ok_or("missing RMVPE decoder levels")?;
        let model = RmvpeDecoder::load(&checkpoint, levels.len(), [2, 2], &device)?;
        let mut maximum = 0.0_f32;
        let outputs = model.forward_with_levels(Tensor::from_data(TensorData::new(input, input_shape), &device), &skips);
        for (index, (tensor, expected)) in outputs.into_iter().zip(levels).enumerate() {
            let shape: [usize; 4] = serde_json::from_value(expected["shape"].clone()).map_err(|e| e.to_string())?;
            if tensor.dims() != shape { return Err(format!("RMVPE Decoder stage {index}: shape mismatch")); }
            let values: Vec<f32> = serde_json::from_value(expected["output"].clone()).map_err(|e| e.to_string())?;
            let actual = tensor.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
            let error = max_abs_error(&actual, &values);
            println!("backend={backend} rmvpe_decoder_stage={index} max_abs={error:.9}");
            maximum = maximum.max(error);
            if !error.is_finite() || error > 1e-4 { return Err(format!("RMVPE Decoder stage {index}: Torch/Burn parity exceeded tolerance")); }
        }
        println!("backend={backend} rmvpe_decoder_stages={} max_abs_rmvpe_decoder={maximum:.9}", model.stages());
        return Ok(());
    }
    if command == "verify-rmvpe-intermediate" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_unet::RmvpeIntermediate};
        let checkpoint = RvcCheckpoint::open_root(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing RMVPE Intermediate reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(fixture_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let shape: [usize; 4] = serde_json::from_value(fixture["input_shape"].clone()).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).map_err(|e| e.to_string())?;
        let expected = fixture["levels"].as_array().ok_or("missing RMVPE Intermediate levels")?;
        let model = RmvpeIntermediate::load(&checkpoint, expected.len(), &device)?;
        let values = model.forward_with_levels(Tensor::from_data(TensorData::new(input, shape), &device));
        let mut maximum = 0.0_f32;
        for (level, (tensor, reference)) in values.into_iter().zip(expected).enumerate() {
            let output_shape: [usize; 4] = serde_json::from_value(reference["shape"].clone()).map_err(|e| e.to_string())?;
            if tensor.dims() != output_shape { return Err(format!("RMVPE Intermediate level {level}: shape mismatch")); }
            let reference_output: Vec<f32> = serde_json::from_value(reference["output"].clone()).map_err(|e| e.to_string())?;
            let actual = tensor.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
            let error = max_abs_error(&actual, &reference_output);
            maximum = maximum.max(error);
            println!("backend={backend} rmvpe_intermediate_level={level} max_abs={error:.9}");
            if !error.is_finite() || error > 1e-4 { return Err(format!("RMVPE Intermediate level {level}: Torch/Burn parity exceeded tolerance")); }
        }
        println!("backend={backend} rmvpe_intermediate_levels={} max_abs_rmvpe_intermediate={maximum:.9}", model.levels());
        return Ok(());
    }
    if command == "verify-rmvpe-encoder" {
        use burn::tensor::{Tensor, TensorData};
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, rmvpe_unet::RmvpeEncoderStage};
        let checkpoint = RvcCheckpoint::open_root(&path)?;
        let reference_path = flow_fixture_path.as_ref().ok_or("missing RMVPE U-Net reference")?;
        let fixture: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(reference_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let input_shape: [usize; 4] = serde_json::from_value(fixture["input_shape"].clone()).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).map_err(|e| e.to_string())?;
        let stages = fixture["stages"].as_array().ok_or("missing U-Net stages")?;
        if stages.is_empty() || stages.len() > 5 { return Err("invalid RMVPE encoder stages".into()); }
        let mut current = Tensor::from_data(TensorData::new(input, input_shape), &device);
        let mut maximum = 0.0_f32;
        for (index, expected) in stages.iter().enumerate() {
            let model = RmvpeEncoderStage::load(&checkpoint, index, &device)?;
            let result = model.forward(current);
            let skip_shape: [usize; 4] = serde_json::from_value(expected["skip_shape"].clone()).map_err(|e| e.to_string())?;
            let pooled_shape: [usize; 4] = serde_json::from_value(expected["pooled_shape"].clone()).map_err(|e| e.to_string())?;
            if result.skip.dims() != skip_shape || result.pooled.dims() != pooled_shape {
                return Err(format!("RMVPE encoder stage {index} shape mismatch"));
            }
            let read = |field: &str| -> Result<Vec<f32>, String> { serde_json::from_value(expected[field].clone()).map_err(|e| e.to_string()) };
            let error_skip = max_abs_error(&result.skip.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?, &read("skip")?);
            let error_pooled = max_abs_error(&result.pooled.clone().into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?, &read("pooled")?);
            maximum = maximum.max(error_skip).max(error_pooled);
            println!("backend={backend} rmvpe_encoder_stage={index} blocks={} max_abs_skip={error_skip:.9} max_abs_pooled={error_pooled:.9}", model.blocks());
            if !error_skip.is_finite() || !error_pooled.is_finite() || error_skip > 1e-4 || error_pooled > 1e-4 {
                return Err(format!("RMVPE U-Net stage {index} Torch/Burn parity exceeded tolerance"));
            }
            current = result.pooled;
        }
        println!("backend={backend} rmvpe_encoder_stages={} max_abs_rmvpe_encoder={maximum:.9}", stages.len());
        return Ok(());
    }
    if command == "verify-rmvpe-head" {
        use burn::tensor::{Tensor, TensorData};
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open_root(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing RMVPE reference")?;
        let fixture: serde_json::Value = serde_json::from_str(&fs::read_to_string(fixture_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let input: Vec<f32> = serde_json::from_value(fixture["features"].clone()).map_err(|e| e.to_string())?;
        let shape: [usize; 3] = serde_json::from_value(fixture["shape"].clone()).map_err(|e| e.to_string())?;
        let expected: Vec<f32> = serde_json::from_value(fixture["expected_salience"].clone()).map_err(|e| e.to_string())?;
        let expected_f0: Vec<f32> = serde_json::from_value(fixture["expected_f0"].clone()).map_err(|e| e.to_string())?;
        let model = tts_with_rvc_rs::rmvpe::RmvpeHead::load(&checkpoint, &device)?;
        let actual = model.forward(Tensor::from_data(TensorData::new(input, shape), &device))
            .into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &expected);
        let f0 = tts_with_rvc_rs::rmvpe::salience_to_f0(&actual, 0.03)?;
        let f0_error = max_abs_error(&f0, &expected_f0);
        println!("backend={backend} max_abs_rmvpe_salience={error:.9} max_abs_f0={f0_error:.9} tensor_count={}", checkpoint.tensor_count());
        if !error.is_finite() || error > 1e-4 || !f0_error.is_finite() || f0_error > 1e-3 {
            return Err("RMVPE output head / F0 parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-nsf-source" {
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, nsf_source::NsfSourceFixture};
        let checkpoint = RvcCheckpoint::open(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing NSF source fixture")?;
        let fixture: NsfSourceFixture = serde_json::from_str(
            &fs::read_to_string(fixture_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let (sines, voiced, noise, merged) = fixture.run(&checkpoint, &device)?;
        println!("backend={backend} max_abs_nsf_sines={sines:.9} max_abs_nsf_uv={voiced:.9} max_abs_nsf_noise={noise:.9} max_abs_nsf_merged={merged:.9}");
        if [sines, voiced, noise, merged].iter().any(|v| !v.is_finite() || *v > 1e-4) {
            return Err("Torch/Burn NSF SineGen or SourceModule parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-nsf-from-f0" {
        use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, nsf_source::NsfFromF0Fixture};
        let checkpoint = RvcCheckpoint::open(&path)?;
        let fixture_path = flow_fixture_path.as_ref().ok_or("missing NSF waveform fixture")?;
        let fixture: NsfFromF0Fixture = serde_json::from_str(
            &fs::read_to_string(fixture_path).map_err(|error| error.to_string())?,
        ).map_err(|error| error.to_string())?;
        let deviation = fixture.run(&checkpoint, &device)?;
        println!("backend={backend} max_abs_nsf_from_f0={deviation:.9}");
        if !deviation.is_finite() || deviation > 1e-4 {
            return Err("Torch/Burn NSF F0→waveform parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-nsf-decoder" {
        let reference_path = flow_fixture_path.as_ref().ok_or("missing NSF decoder reference")?;
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open(&path)?;
        let data = fs::read_to_string(reference_path).map_err(|error| error.to_string())?;
        let reference: tts_with_rvc_rs::decoder::NsfDecoderFixture = serde_json::from_str(&data).map_err(|error| error.to_string())?;
        let decoder = reference.load(&checkpoint, &device)?;
        let (input, harmonic, g) = reference.tensors(&device);
        let actual = decoder.forward(input, harmonic, g).into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let error = max_abs_error(&actual, &reference.expected_waveform);
        println!("backend={backend} stages={} max_abs_waveform={error:.9}", decoder.stage_count());
        if !error.is_finite() || error > 1e-4 {
            return Err("Torch/Burn NSF waveform decoder parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-decoder-stage" {
        let reference_path = flow_fixture_path.as_ref().ok_or("missing decoder reference")?;
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open(&path)?;
        let data = fs::read_to_string(reference_path).map_err(|error| error.to_string())?;
        let reference: tts_with_rvc_rs::decoder::DecoderStageFixture = serde_json::from_str(&data).map_err(|error| error.to_string())?;
        let stage = reference.load(&checkpoint, &device)?;
        let (input, source) = reference.tensors(&device);
        let upsampled = stage.upsample_only(input.clone()).into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let up_error = max_abs_error(&upsampled, &reference.expected_upsampled);
        let result = stage.forward(input, source).into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let error = max_abs_error(&result, &reference.expected_output);
        println!("backend={backend} stage={} resblocks={} max_abs_upsample={up_error:.9} max_abs_decoder_stage={error:.9}", reference.stage, stage.residual_count());
        if !up_error.is_finite() || up_error > 1e-4 || !error.is_finite() || error > 1e-4 {
            return Err("Torch/Burn NSF decoder stage parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-flow" {
        let flow_fixture_path = flow_fixture_path.as_ref().ok_or("missing flow reference")?;
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open(&path)?;
        let data = fs::read_to_string(flow_fixture_path).map_err(|error| error.to_string())?;
        let fixture: tts_with_rvc_rs::flow::FlowFixture = serde_json::from_str(&data).map_err(|error| error.to_string())?;
        let flow = tts_with_rvc_rs::flow::ResidualCouplingBlock::load(&checkpoint, &device)?;
        let (input, mask, condition) = fixture.tensors(&device);
        let forward = flow.forward(input, mask.clone(), Some(condition.clone()), false);
        let actual = forward.clone().into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let error = max_abs_error(&actual, &fixture.expected_forward);
        let reverse = flow.forward(forward, mask, Some(condition), true);
        let actual = reverse.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        let reverse_error = max_abs_error(&actual, &fixture.expected_reverse);
        println!("backend={backend} flow_layers={} max_abs_flow_forward={error:.9} max_abs_flow_reverse={reverse_error:.9}", flow.len());
        if !error.is_finite() || error > 1e-4 || !reverse_error.is_finite() || reverse_error > 1e-4 {
            return Err("Torch/Burn ResidualCouplingBlock parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-text-encoder-pth" {
        use burn::tensor::{Int, Tensor, TensorData};
        let checkpoint = tts_with_rvc_rs::checkpoint::RvcCheckpoint::open(&path)?;
        let reference_path = flow_fixture_path.as_ref().ok_or("missing TextEncoder reference")?;
        let data = fs::read_to_string(reference_path).map_err(|error| error.to_string())?;
        let fixture: TextEncoderFixture = serde_json::from_str(&data).map_err(|error| error.to_string())?;
        let model = tts_with_rvc_rs::text_encoder::TextEncoder::load(&checkpoint, &device)?;
        let (batch, time, channels) = (fixture.front.batch, fixture.front.time, fixture.front.in_channels);
        let phone = Tensor::from_data(TensorData::new(fixture.front.phone.clone(), [batch, time, channels]), &device);
        let pitch = Tensor::<2, Int>::from_data(TensorData::new(fixture.front.pitch.clone(), [batch, time]), &device);
        let verify = |out: tts_with_rvc_rs::text_encoder::TextEncoderOutput, mean: &[f32], logs: &[f32], mask: &[f32]| -> Result<f32, String> {
            let actual_mean = out.mean.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
            let actual_logs = out.log_scale.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
            let actual_mask = out.mask.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
            let deviation = max_abs_error(&actual_mean, mean).max(max_abs_error(&actual_logs, logs));
            let mask_error = max_abs_error(&actual_mask, mask);
            if !deviation.is_finite() || deviation > 1e-4 || mask_error != 0.0 {
                return Err(format!("native TextEncoder parity failed: deviation={deviation}, mask={mask_error}"));
            }
            Ok(deviation)
        };
        let forward = model.forward(phone.clone(), Some(pitch.clone()), &fixture.front.lengths, None);
        let error = verify(forward, &fixture.expected_mean, &fixture.expected_log_scale, &fixture.expected_mask)?;
        let skipped = model.forward(phone, Some(pitch), &fixture.front.lengths, Some(fixture.skip_head));
        let skip_error = verify(skipped, &fixture.expected_skip_mean, &fixture.expected_skip_log_scale, &fixture.expected_skip_mask)?;
        println!("backend={backend} max_abs_native_text_encoder={error:.9} max_abs_native_skip_head={skip_error:.9}");
        return Ok(());
    }
    let data = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    if command == "verify-text-encoder" {
        let fixture: TextEncoderFixture = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        let output = fixture.forward(&device, None);
        let mean = output.mean.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let logs = output.log_scale.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let mask = output.mask.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&mean, &fixture.expected_mean)
            .max(max_abs_error(&logs, &fixture.expected_log_scale));
        let mask_error = max_abs_error(&mask, &fixture.expected_mask);
        println!("backend={backend} max_abs_text_encoder={error:.9} max_abs_mask={mask_error:.9}");
        if !error.is_finite() || error > 1e-4 || mask_error > 0.0 {
            return Err("Torch/Burn full TextEncoder parity exceeded tolerance".into());
        }
        let skipped = fixture.forward(&device, Some(fixture.skip_head));
        let mean = skipped.mean.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let logs = skipped.log_scale.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let mask = skipped.mask.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let error = max_abs_error(&mean, &fixture.expected_skip_mean)
            .max(max_abs_error(&logs, &fixture.expected_skip_log_scale));
        let mask_error = max_abs_error(&mask, &fixture.expected_skip_mask);
        println!("backend={backend} max_abs_skip_head={error:.9} max_abs_skip_mask={mask_error:.9}");
        if !error.is_finite() || error > 1e-4 || mask_error > 0.0 {
            return Err("Torch/Burn TextEncoder skip_head parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-encoder-layer" {
        let fixture: EncoderLayerFixture = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        let actual = fixture.forward(&device).into_data().try_into_vec::<f32>()
            .map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &fixture.expected_encoder);
        println!("backend={backend} max_abs_encoder_layer={error:.9}");
        if !error.is_finite() || error > 1e-4 {
            return Err("Torch/Burn encoder layer parity exceeded tolerance".into());
        }
        return Ok(());
    }
    if command == "verify-attention" {
        let fixture: AttentionFixture = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        let actual = fixture.forward(&device).into_data().try_into_vec::<f32>()
            .map_err(|e| e.to_string())?;
        let error = max_abs_error(&actual, &fixture.expected);
        println!("backend={backend} max_abs_attention={error:.9}");
        if !error.is_finite() || error > 1e-4 {
            return Err("Torch/Burn attention parity exceeded tolerance".into());
        }
        return Ok(());
    }
    let fixture: TextFrontFixture = serde_json::from_str(&data).map_err(|e| e.to_string())?;
    let result = fixture.forward(&device);
    let actual_features = result.features.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
    let actual_mask = result.mask.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
    let feature_error = max_abs_error(&actual_features, &fixture.expected_features);
    let mask_error = max_abs_error(&actual_mask, &fixture.expected_mask);
    println!("backend={backend} max_abs_features={feature_error:.9} max_abs_mask={mask_error:.9}");
    if !feature_error.is_finite() || feature_error > 1e-4 || mask_error > 0.0 {
        return Err("Torch/Burn parity exceeded tolerance".into());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
