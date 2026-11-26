# Stage 33 next steps — NSF CPU scheduling and end-to-end validation

Repository: `Atm4x/Vendor-for-burn`. Start from the local `experiment/nsf-universal-polyphase-20261109` head after Stage32. Use the Stage32 report in `models/tts-with-rvc-rs/docs/STAGE32_NSF_PARALLEL_AB.md`.

Constraints: one Cargo process at a time, `CARGO_BUILD_JOBS=2`, no `cargo clean`, offline locked, reuse `/mnt/data/stage22-work/target`. Keep private `CrazyMita.pth`, `hubert_base.pt`, `rmvpe.pt`, `CrazyMita.index`, `MitaKind.wav` outside Git. Do not modify unrelated legacy scripts or preexisting untracked Stage27/29 benchmarks.

Stage32 delivered: deterministic parallel residual branches; benchmark mode switch in one binary; serial control matching original accumulation; 15/15 release unit + 31/31 release parity; real 10.1s RVC output bitwise equal to Stage31. A/B 24frames: 1.38x (2 threads), 1.60x (4 threads); 128frames: 0.99x (2 threads), 1.29x (4 threads). The short A/B is insufficient to infer a guaranteed speedup on all CPU loads. Full 90s validation not completed.

Priorities: profile nested GEMM Rayon contention and fine-grained scheduler costs in ResBlocks; use a shared one-binary alternating A/B with exact same real latent inputs. Then add resumable chunk checkpointing so full RVC 89.94s is not lost on a timeout. Only after stable parity and full WAV check consider a separate PR; use GitHub connector directly and do not touch main.
