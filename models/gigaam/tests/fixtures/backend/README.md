# Synthetic CubeCL backend smoke fixtures

All four Safetensors checkpoints and the short 16 kHz WAV are synthetic test assets, with small dimensions (`d_model=8`, one Conformer block). They were generated with a fixed NumPy random seed using the Rust loader's exact tensor names and shapes. The file `ctc/model.safetensors` contains 2,044 scalar parameters; SSL shares its encoder, while Emo and RNN-T provide their respective heads.

They are not official GigaAM checkpoints and do not produce meaningful speech recognition. Their purpose is to test the **same forward path** for Conformer/CTC, Conformer/LSTM/RNN-T, Conformer/SSL, and Conformer/Emo under CubeCL CPU and optional Flex without downloading any model or requiring Python to run the tests.

The source audio is a synthetic 0.4-second sine wave, not user speech. No downloaded weights, copyrighted recordings or external model assets are included.
