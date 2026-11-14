# Stage 94 — cost-aware global chunk execution, now with shape-aware tensor batches

This experiment does not alter the per-chunk model inputs, weights or source random field.
`RVC_BATCH_SCHEDULE=shape-gather` groups *nonadjacent but exactly equal-length* chunks into
one tensor batch, with original IDs and chronological PCM assembly. No padding is inserted.
Default `RVC_BATCH_SCHEDULE` remains `contiguous` for the established strategies.

`RVC_CHUNK_STRATEGY=batch-aware` jointly chooses a whole-recording length lattice under the
Stage93 analytical work proxy, charging the dispatch/launch term for actual shape-gather
batches. This strategy enables shape-gather by default; `RVC_BATCH_LOOKAHEAD=2..=256`
controls the gather horizon (default 64). Other strategies are not affected.

The global lattice optimizer is deliberately a restricted search plus the full Stage93
shortest-path candidate. It guarantees a score no worse than Stage93 *under its own
uncalibrated work proxy* but is not a proof of optimal GPU latency, and may move quiet
boundaries into audible passages. Distinct plans are NOT guaranteed to produce the same WAV.

Compile and run the standalone geometry experiment:

`rustc --edition=2024 -O research/stage94-batch-dag/measure_real.rs -o stage94_measure`

`./stage94_measure MitaKind.wav > stage94_real_wav.jsonl`

The runner `Run-RVC-Full-Stage94.ps1` performs an A/B/C/C/B/A test with the same
model files under `embed`, comparing consecutive batches, gather-batching on the same
plan, and batch-aware planning plus gather. All three cases enable global F0, phase,
noise; mixed FP16 remains off. The new Windows executable must be built from Stage94.
