"""Convert vocal effort by swapping LombardTokenizer VQ2 codes."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--source", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import torch
    from lombardtokenizer.inference import load_audio, load_tokenizer, save_audio

    model = load_tokenizer(args.checkpoint, args.config, args.device)
    source = load_audio(args.source, model.sample_rate, args.device)
    reference = load_audio(args.reference, model.sample_rate, args.device)
    with torch.inference_mode():
        converted = model.convert_vocal_effort(source, reference)
    save_audio(args.output, converted, model.sample_rate)


if __name__ == "__main__":
    main()
