from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BERT_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
DEFAULT_INPUT_PATH = Path(__file__).with_name("synthea_diabetes_training_updated.parquet")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("vocab_synthea.txt")


def iter_tokens(value) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(token) for token in value.tolist() if token is not None]
    if isinstance(value, (list, tuple)):
        return [str(token) for token in value if token is not None]
    return []


def build_vocab(parquet_path: Path, token_column: str = "sorted_event_tokens") -> list[str]:
    df = pd.read_parquet(parquet_path, columns=[token_column])
    counter: Counter[str] = Counter()

    for seq in df[token_column]:
        counter.update(iter_tokens(seq))

    tokens = sorted(counter, key=lambda token: (-counter[token], token))
    return BERT_SPECIAL_TOKENS + tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a BERT vocabulary text file from a parquet dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the parquet dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output vocab text file.",
    )
    parser.add_argument(
        "--token-column",
        default="sorted_event_tokens",
        help="Column containing token sequences.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab = build_vocab(args.input, token_column=args.token_column)
    args.output.write_text("\n".join(vocab), encoding="utf-8")
    print(f"Wrote {len(vocab)} tokens to {args.output}")


if __name__ == "__main__":
    main()
