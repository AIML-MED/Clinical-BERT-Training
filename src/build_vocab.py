import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

BERT_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def iter_tokens(value):
    if isinstance(value, np.ndarray):
        return [str(token) for token in value.tolist() if token is not None]
    if isinstance(value, (list, tuple)):
        return [str(token) for token in value if token is not None]
    return []


def build_vocab(parquet_path):
    df = pd.read_parquet(parquet_path, columns=["sorted_event_tokens"])
    counter = Counter()

    for seq in df["sorted_event_tokens"]:
        counter.update(iter_tokens(seq))

    tokens = sorted(counter, key=lambda t: (-counter[t], t))
    return BERT_SPECIAL_TOKENS + tokens


def main():
    input_path = Path("datasets/bert/synthea_diabetes_training.parquet")
    output_path = Path("datasets/bert/vocab.txt")

    vocab = build_vocab(input_path)
    output_path.write_text("\n".join(vocab))

    print(f"Vocab size: {len(vocab)}")


if __name__ == "__main__":
    main()
