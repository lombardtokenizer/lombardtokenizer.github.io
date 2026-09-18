"""Train LT2/E2 independently from LombardTokenizer."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    from lombardtokenizer.training import run_vocal_effort_training

    run_vocal_effort_training(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
