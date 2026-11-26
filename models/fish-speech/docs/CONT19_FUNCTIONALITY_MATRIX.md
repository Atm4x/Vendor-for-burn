# Fish Speech Rust — CONT18 functional coverage (original Python 1.5)

**Metric preserved from CONT17:** (15 DONE + 7 × 0.5 PARTIAL)/22 = **84.09%**. It measures implemented source features on Rust, excludes Windows/CUDA validation gates. It does not assign proportional complexity weights. CONT18 advances F22 within PARTIAL; no milestone is relabelled DONE without its remaining production implementation.

| ID | Python functionality | State | Specific remaining functionality |
|---|---|---|---|
| F01 | Semantic + Firefly configuration | PARTIAL | Complete Firefly variants and verified checkpoint aliases; strict source hashes exist |
| F02 | SafeTensors conversion/loading | DONE | Real checkpoint parity is an independent verification gate |
| F03 | Fish BPE tokenizer | DONE | Golden check on authentic tokenizer is independent verification |
| F04 | `clean_text` | DONE | — |
| F05 | `split_text` | DONE | — |
| F06 | Conversation/VQ serialization | DONE | — |
| F07 | Reference/no-reference prompts | DONE | — |
| F08 | `ReferenceLoader.load_audio()` | PARTIAL | MP3/AAC/M4A/WMA offline decoding dependencies |
| F09 | Resampler and LogMel | DONE | — |
| F10 | Weighted Firefly encode | DONE | Authentic-weight parity is independent verification |
| F11 | Slow AR prefill | PARTIAL | Execute real TensorRT engine, with correct external device inputs |
| F12 | Slow cached decode | PARTIAL | CUDA KV writes, active-prefix attention and same-stream execution |
| F13 | Fast AR generation | PARTIAL | GPU codebook loop, GPU sampling and Fast KV state ownership |
| F14 | Naive AR generation | PARTIAL | Native TensorRT production execution |
| F15 | Sampling, repetition, EOS | DONE | GPU sampling integration remains under F13 |
| F16 | Long text generation | DONE | — |
| F17 | Weighted Firefly decoder | DONE | Authentic voice parity is independent verification |
| F18 | 44.1kHz PCM, segment merge | DONE | — |
| F19 | Complete CPU TTS assembly | DONE | Full synthetic two-checkpoint synthesis verified CONT17, not re-run CONT18 |
| F20 | Header/segment/final/error events | DONE | — |
| F21 | Reference cache by ID/hash | DONE | — |
| F22 | Cancellation / queue / model lifecycle | **PARTIAL; advanced CONT18** | CUDA stream/device allocation lifecycle, true GPU warmup and reusable bounded GPU pool. Now implemented: per-token cancel, atomic reload, close, bounded FIFO and request payloads, active cancel, scratch admission/retirement, panic fault isolation |

## Separate operability gates

| Verification | CONT18 status |
|---|---|
| Full Core test suites | PASS, 218 tests; 2 ignored |
| Outer crate tests | PASS, 29 tests; 2 ignored |
| Python checkpoint exporter | PASS, 5/5 |
| Rust formatting | PASS |
| TensorRT 1.7 API compilation | PASS, `cargo check --locked --offline --features tensorrt-api` |
| Full trained checkpoint and tokenizer parity | BLOCKED_ON_ARTIFACT |
| Real Windows NVIDIA TensorRT-RTX inference | NOT_TESTED_GPU |
| Full GPU TTS, numerical parity and perf | NOT_TESTED_GPU |
| Windows packaging, CUDA DLL and runtime bundle | NOT_TESTED |

Remaining functionally heavy areas F11–F14 require the actual TensorRT runtime and GPU-stream execution implementation. The ability to run these operations on Windows is a **separate** validation gate and does not itself change the functional count.

## CONT19 update

**Functional score unchanged: 84.1%.** New real TensorRT Builder/bindings/enqueue code and CUDA Driver KV-copy code advance F11–F13 and F22 but none of those seven PARTIAL features is complete. Real Windows GPU verification remains a separate operability gate; local compiler success is not GPU execution.

**Publication:** GitHub commit `342f1b361f03f448cf2d704b66206eee531513a1` publishes a subset (engine builder contract) on the older remote branch. The full local CONT19 source is in the ZIP, and all previous local cycles CONT6–CONT18 still require GitHub synchronization.
