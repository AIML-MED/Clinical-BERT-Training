import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic 1:1 case-control test set.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-column", default="COHORT")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_parquet(args.input)
    counts = data[args.target_column].value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError(f"Expected binary labels 0 and 1, found {sorted(counts.index)}")

    sample_size = int(counts.min())
    balanced = pd.concat(
        [
            data[data[args.target_column] == label].sample(
                n=sample_size,
                random_state=args.random_state,
                replace=False,
            )
            for label in (0, 1)
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    balanced.to_parquet(args.output, index=False)
    print(f"Input rows: {len(data)}")
    print(f"Input prevalence: {data[args.target_column].mean():.6f}")
    print(f"Balanced rows: {len(balanced)}")
    print(balanced[args.target_column].value_counts().sort_index().to_string())
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
