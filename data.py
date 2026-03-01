import argparse
import re
from pathlib import Path
from typing import Iterable, List

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a small TinyStories subset from Hugging Face and write it to a local text file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="roneneldan/TinyStories",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional dataset config name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name that contains story text.",
    )
    parser.add_argument(
        "--num-stories",
        type=int,
        default=100,
        help="Number of random stories to write.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used by dataset shuffle.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=10_000,
        help="Shuffle buffer size used for streaming randomization.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stories.txt",
        help="Output text file path.",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="\n\n",
        help="Story separator written between sampled stories (default: blank line).",
    )
    parser.add_argument(
        "--no-clean-mojibake",
        action="store_true",
        help="Disable mojibake cleanup (enabled by default).",
    )
    return parser.parse_args()


def sample_stories(stream: Iterable[dict], text_column: str, num_stories: int) -> List[str]:
    stories: List[str] = []
    for row in stream:
        text = row.get(text_column)
        if not isinstance(text, str):
            continue
        story = text.strip()
        if not story:
            continue
        stories.append(story)
        if len(stories) >= num_stories:
            break
    return stories


MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "Â ": " ",
    "Â": "",
}

# Common residual mojibake glyphs seen in corrupted UTF-8/CP1252 text.
MOJIBAKE_CHARS_RE = re.compile(r"[â€™€™œ¦“”]")


def clean_mojibake(text: str) -> str:
    cleaned = text
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, good)
    # Drop remaining mojibake singleton glyphs and collapse whitespace.
    cleaned = MOJIBAKE_CHARS_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    return cleaned.strip()


def main() -> None:
    args = parse_args()
    if args.num_stories <= 0:
        raise ValueError("--num-stories must be > 0")

    dataset = load_dataset(
        path=args.dataset,
        name=args.config,
        split=args.split,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=max(1, args.shuffle_buffer))

    stories = sample_stories(dataset, args.text_column, args.num_stories)
    if len(stories) < args.num_stories:
        raise RuntimeError(
            f"Only collected {len(stories)} stories from {args.dataset}/{args.split}; expected {args.num_stories}."
        )

    cleaned_count = 0
    if not args.no_clean_mojibake:
        cleaned_stories: List[str] = []
        for story in stories:
            cleaned = clean_mojibake(story)
            if cleaned != story:
                cleaned_count += 1
            if cleaned:
                cleaned_stories.append(cleaned)
        stories = cleaned_stories

    output_path = Path(args.output)
    output_path.write_text(args.separator.join(stories) + "\n", encoding="utf-8")

    chars = sum(len(s) for s in stories)
    print(f"Wrote {len(stories)} stories ({chars} chars) to {output_path}")
    if not args.no_clean_mojibake:
        print(f"Mojibake cleanup touched {cleaned_count} stories.")
    print(f"Dataset: {args.dataset} | split: {args.split} | seed: {args.seed}")


if __name__ == "__main__":
    main()
