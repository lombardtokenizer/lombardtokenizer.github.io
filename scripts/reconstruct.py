"""Reconstruct one waveform with LombardTokenizer."""

from __future__ import annotations

import argparse


def load_waveform(path: str, sample_rate: int):
    from lombardtokenizer.inference import load_audio

    return load_audio(path, sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import torch
    from lombardtokenizer.inference import load_tokenizer, save_audio

    model = load_tokenizer(args.checkpoint, args.config, args.device)
    waveform = load_waveform(args.input, model.sample_rate).to(args.device)
    with torch.inference_mode():
        reconstructed = model.reconstruct(waveform)
    save_audio(args.output, reconstructed, model.sample_rate)


if __name__ == "__main__":
    main()
