"""Capture all five V5 Russian speakers' managed-duration mel outputs."""

import argparse
import json
import warnings
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.package import PackageImporter


def run(checkpoint: Path, output: Path) -> None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    package = PackageImporter(str(checkpoint)).load_pickle("tts_models", "model").packages[0]
    package.unpack_q_model()
    source = json.loads((output / "symbol-durations-parity.json").read_text(encoding="utf-8"))[0]
    ids = torch.tensor([source["ids"]], dtype=torch.long)
    rates = torch.tensor([source["rates"]], dtype=torch.float)
    pitches = torch.tensor([source["pitches"]], dtype=torch.float)
    pauses = {int(key): value for key, value in source["pauses"].items()}
    output_mel = {}
    manifest = []
    for speaker, speaker_id in sorted(package.speaker_to_ids[0].items(), key=lambda x: x[1]):
        sid = torch.tensor([speaker_id], dtype=torch.long)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wave, durations = package.models[0](sequence=ids, speaker_ids=sid, sr=8000,
                symb_durs=pauses, durs_rate=rates, pitch_coefs=pitches, device="cpu")
            mask = torch.zeros_like(ids, dtype=torch.bool)
            raw_pitch = package.models[0].pitch_predictor(ids, sid, mask, 1.0)
            f0 = package.models[0].update_pitch_coef(raw_pitch, pitches, sid)
            mel = package.models[0].tacotron(ids, sid, mask, durations, f0, None)
            if pauses:
                mel = package.models[0].fx_pauses(mel, durations, pauses)
        output_mel[speaker] = mel.detach().cpu().float().contiguous()
        manifest.append({"speaker": speaker, "speaker_id": speaker_id,
                         "durations": durations.cpu().float().flatten().tolist(),
                         "mel_shape": list(output_mel[speaker].shape)})
        print(f"speaker {speaker} id={speaker_id} frames={mel.shape[-1]}", flush=True)
    (output / "multispeaker-ssml-parity.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save_file(output_mel, str(output / "multispeaker-ssml-parity.safetensors"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
