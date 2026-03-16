import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.tokenizer_utils import train_bpe_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/stories.txt")
    parser.add_argument("--output", type=str, default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    if args.vocab_size < 128:
        raise ValueError("--vocab-size should be >= 128 for stable BPE behavior.")
    if args.min_frequency < 1:
        raise ValueError("--min-frequency must be >= 1")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    tokenizer = train_bpe_tokenizer(
        input_path=args.input,
        tokenizer_path=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    vocab_size = tokenizer.get_vocab_size()
    print(f"saved tokenizer to {args.output} (vocab_size={vocab_size})")


if __name__ == "__main__":
    main()
