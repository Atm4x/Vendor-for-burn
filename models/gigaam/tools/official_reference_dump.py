"""Export original GigaAM Python tensors to compare with native Burn inference."""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream.resolve()))
    import numpy as np
    import torch
    import gigaam
    from gigaam.model import GigaAM, GigaAMASR, GigaAMEmo

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["cfg"]
    model_name = args.model_name or args.checkpoint.stem
    config.model_name = model_name
    config.encoder.flash_attn = False
    if args.tokenizer is not None:
        config.decoding.model_path = str(args.tokenizer.resolve())
    elif "e2e" in model_name or model_name == "v1_rnnt":
        tokenizer = args.checkpoint.with_name(model_name + "_tokenizer.model")
        if not tokenizer.is_file():
            parser.error(f"Missing tokenizer: {tokenizer}")
        config.decoding.model_path = str(tokenizer.resolve())

    if "ssl" in model_name:
        model = GigaAM(config)
    elif "emo" in model_name:
        model = GigaAMEmo(config)
    else:
        model = GigaAMASR(config)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval().to(torch.device(args.device))
    args.output.mkdir(parents=True, exist_ok=True)

    def dump(name: str, tensor: torch.Tensor) -> list[int]:
        array = tensor.detach().cpu().contiguous().to(dtype=torch.float32).numpy()
        array.astype(np.dtype("<f4"), copy=False).tofile(args.output / f"{name}.f32")
        return list(array.shape)

    with torch.inference_mode():
        wav, length = model.prepare_wav(str(args.audio))
        if isinstance(model, GigaAMASR) and int(length.item()) > 25 * 16_000:
            parser.error("Use an audio snippet <=25 seconds for short-form ASR parity")
        mel, mel_len = model.preprocessor(wav, length)
        encoded, encoded_len = model.forward(wav, length)
        mel_shape = dump("mel", mel[0])
        encoder_shape = dump("encoder", encoded[0].transpose(0, 1))
        meta = {
            "model_name": model_name,
            "device": args.device,
            "audio_samples": int(length.item()),
            "mel_shape": mel_shape,
            "mel_length": int(mel_len.item()),
            "encoder_shape": encoder_shape,
            "encoder_length": int(encoded_len.item()),
            "encoder_layout": "[time, channels]",
            "logits_kind": None,
        }
        if isinstance(model, GigaAMASR):
            text, tokens, token_frames = model.decoding.decode(model.head, encoded, encoded_len)[0]
            meta.update({"text": text, "tokens": tokens, "token_frames": token_frames})
            if "ctc" in model_name:
                raw_logits = model.head.decoder_layers(encoded).transpose(1, 2)[0]
                meta["logits_shape"] = dump("logits", raw_logits)
                meta["logits_kind"] = "pre_log_softmax"
        if isinstance(model, GigaAMEmo):
            probs = torch.softmax(model.head(encoded.mean(dim=-1))[0], dim=-1)
            meta["emotions"] = {str(model.id2name[i]): float(x) for i, x in enumerate(probs.tolist())}

    (args.output / "reference.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
