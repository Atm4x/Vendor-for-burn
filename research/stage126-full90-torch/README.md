# Stage126 — 90-second Rust RVC and original PyTorch CUDA comparison

One Windows CI package includes the Rust 1.100.0 CUDA EXE, complete Stage126 full90 runner and original PyTorch helper, Stage125 NCU/NSYS runner, technical report and workflow contract. No models, WAV or index are bundled.

From the package directory with your existing embed folder:

    .\Run-Stage126-Full90-Compare.ps1 -Runs 3 -Cycles 2

Source MUST actually contain 85–95 seconds of audio (the historical reference WAV was 89.939s). If embed/MitaKind.wav is shorter, use -InputWav D:\recordings\long-89-second.wav. The experiment never truncates or repeats audio. Use -CheckOnly for duration/files preflight. Use -OriginalRoot for a different folder containing original tts_with_rvc/infer/vc/pipeline.py. The ONLY Python interpreter configured is D:\Projects\tts-with-rvc\.venv\Scripts\python.exe.

Rust runs control, linear, residue and combined NTC routes on the full WAV with CUDA, persistent RMVPE GRU, core300/batch2, direct NSF, shared NTC and tile4. Each mode runs 3 complete full-waveform passes in a fresh model process, excluding the first pass from the warm median. Two cycles reverse the process ordering. Route PCM16 must be bit-identical to the Rust control or the comparison is rejected. Optional legacy mode is diagnostic only.

The original PyTorch CUDA comparison uses a separate local trusted Python source tree, not modified by the runner. Its FP32 baseline disables matmul and cuDNN TF32, and its FP16 baseline follows the original pipeline. Original model and index are the same physical files as the Rust run. The five Stage80 original source SHA256 fingerprints are recorded and mismatches are labelled.

Important scope difference: Rust full benchmark times synchronized resident neural inference, excluding input preprocessing and final postprocessing; original PyTorch VC.pipeline.pipeline includes its filtering and normalization. RVC chunking, F0, and FP32 math differ. Cross-runtime ratios are full-product engineering throughput observations, NOT same-graph or kernel speed comparisons; Rust-to-Rust PCM checks are the strict acceptance gate.

One stage126_full90_results/UTC.zip contains benchmarks, source hashes, PCM comparisons, GPU identity, logs and metadata, with no WAV/checkpoints/FAISS weights. WAV can be kept locally with -KeepWav, never archived. 90s latency measurements run without NCU/NSYS to avoid profiler replay distortion; the included Stage125 runner handles representative 3s GPU kernel and full NSYS SQLite profiling.
