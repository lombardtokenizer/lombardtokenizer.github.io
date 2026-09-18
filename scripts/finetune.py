"""Fine-tune LombardTokenizer on AVID from a structured YAML config."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="pretraining model checkpoint")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.checkpoint is None and args.resume is None:
        parser.error("one of --checkpoint or --resume is required")
    if args.checkpoint is not None and args.resume is not None:
        parser.error("--checkpoint and --resume cannot be used together")
    from lombardtokenizer.training import run_tokenizer_training

    run_tokenizer_training(
        args.config,
        mode="finetune",
        resume=args.resume,
        fine_tune=args.checkpoint,
    )


if __name__ == "__main__":
    main()
