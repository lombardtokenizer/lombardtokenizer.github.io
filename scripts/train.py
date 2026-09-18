"""Train LombardTokenizer from a structured YAML configuration."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    from lombardtokenizer.training import run_tokenizer_training

    run_tokenizer_training(args.config, mode="pretrain", resume=args.resume)


if __name__ == "__main__":
    main()
