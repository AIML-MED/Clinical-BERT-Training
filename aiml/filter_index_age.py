import argparse
from pathlib import Path

import pandas as pd


def filter_index_age(input_path: Path, output_path: Path, audit_path: Path, min_age: float, max_age: float) -> None:
    df = pd.read_parquet(input_path)
    if "age_at_index_years" not in df.columns:
        raise ValueError(f"{input_path} does not contain age_at_index_years")

    keep = df["age_at_index_years"].between(min_age, max_age, inclusive="both")
    audit = (
        df.assign(keep_age_eligible=keep)
        .groupby(["COHORT", "keep_age_eligible"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[keep].to_parquet(output_path, index=False)
    audit.to_csv(audit_path, index=False)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Audit: {audit_path}")
    print(f"Rows before: {len(df):,}")
    print(f"Rows after: {int(keep.sum()):,}")
    print(f"Rows removed: {int((~keep).sum()):,}")
    print(audit.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a cohort by age_at_index_years.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--min-age", type=float, default=5.0)
    parser.add_argument("--max-age", type=float, default=50.0)
    args = parser.parse_args()
    filter_index_age(args.input, args.output, args.audit, args.min_age, args.max_age)


if __name__ == "__main__":
    main()
