"""Extract offline mHuBERT features for a directory of waveforms."""

import argparse
from pathlib import Path


def _layer(value: str):
    return value if value == "avg" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="utter-project/mHuBERT-147")
    parser.add_argument("--processor")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", required=True, type=_layer)
    parser.add_argument(
        "--extensions",
        default="wav,flac",
        help="comma-separated audio extensions to extract (default: wav,flac)",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import numpy as np
    import torch
    import torchaudio

    from lombardtokenizer.semantic.mhubert import MHuBERTTeacher

    teacher = MHuBERTTeacher(
        args.model,
        feature_extractor_path=args.processor,
        layer=args.layer,
        sample_rate=args.sample_rate,
        device=args.device,
    )
    source_root = Path(args.audio_dir)
    output_root = Path(args.output_dir)
    extensions = {
        "." + extension.strip().lower().lstrip(".")
        for extension in args.extensions.split(",")
        if extension.strip()
    }
    if not extensions:
        raise ValueError("--extensions must contain at least one extension")
    audio_paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    for audio_path in audio_paths:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        embedding = teacher(waveform, sample_rate=sample_rate)
        relative = audio_path.relative_to(source_root).with_suffix(".npy")
        output_path = output_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embedding.squeeze(0).cpu().numpy())


if __name__ == "__main__":
    main()
