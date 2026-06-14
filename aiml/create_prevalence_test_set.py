import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic test set at a target prevalence.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-prevalence", type=float, required=True)
    parser.add_argument("--target-column", default="COHORT")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.target_prevalence < 1:
        raise ValueError("target-prevalence must be between 0 and 1")

    data = pd.read_parquet(args.input)
    controls = data[data[args.target_column] == 0]
    cases = data[data[args.target_column] == 1]
    if len(controls) == 0 or len(cases) == 0:
        raise ValueError("Both controls and cases are required")

    requested_cases = round(
        len(controls) * args.target_prevalence / (1 - args.target_prevalence)
    )
    if requested_cases > len(cases):
        raise ValueError(
            f"Need {requested_cases} cases but only {len(cases)} are available"
        )

    sampled_cases = cases.sample(
        n=requested_cases,
        random_state=args.random_state,
        replace=False,
    )
    output = pd.concat([controls, sampled_cases], ignore_index=True)
    output = output.sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    output.to_parquet(args.output, index=False)

    print(f"Input rows: {len(data)}")
    print(f"Target prevalence: {args.target_prevalence:.6f}")
    print(f"Output rows: {len(output)}")
    print(output[args.target_column].value_counts().sort_index().to_string())
    print(f"Output prevalence: {output[args.target_column].mean():.6f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
