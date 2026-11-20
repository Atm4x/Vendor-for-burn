# F5-TTS Rust — next cycle from CONT12

Continue from the newly published CONT12 branch under `models/f5_tts` only. Work locally and publish at end through GitHub Connector, in a **new** branch based on CONT12. Do not open or modify `maryml`, Fish Speech, RVC, or Legacy. No ONNX. The two model backends remain CubeCL CUDA/WGPU and independent TensorRT-RTX.

Primary metric: **functional coverage and numerical identity with the original Python model**. Secondary metric: actual hardware workability, not `cargo check`. CONT12 estimates 84% functionality / 43% workability; adjust only with evidence.

Completed CONT12: native `jieba-rs` matching 1,809/1,809 `rjieba` inputs, 26,703 pypinyin character readings and 47,111 phrases matching 5,859/5,859 Python pronunciation segments; complete token IDs 1,809/1,809. Offline Rust 46/46, frontend 35/35, two CubeCL CPU full synthetic DiT→CFM→Vocos→PCM parity tests, CUDA/WGPU compile checks. Runtime has no Python dependency.

Do not rebuild the entire environment. Rust 1.100.0 and LLVM 23.1.0 are already available. Use the existing CONT09 offline vendor plus 16 added source directories from `f5_tts_cont12_vendor_additions.tar.zst` with CONT12 `Cargo.lock` (Burn 0.22.1, CubeCL 0.11.3 kept pinned). Or use a fresh verified runner artifact only if necessary.

Priority 1: design model-specific native TensorRT-RTX backend using Rust API/direct TensorRT RTX graph construction, **no ONNX**. Preserve the existing CubeCL data flow and independent warm model/worker lifecycle semantics. Build interfaces with testable contracts, do not fake a working engine. Priority 2: expand numeric parity to production-width DiT forward on synthetic weights. Priority 3: extend Chinese corpus only if it reveals real mismatches. Real pretrained checkpoint validation and Windows GPU execution remain secondary workability gates.
