import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.age_matching import frequency_match_controls_by_age


def main() -> None:
    parser = argparse.ArgumentParser(description="Frequency-match training controls by age.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--controls-per-case", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_parquet(args.input)
    result = frequency_match_controls_by_age(
        data,
        controls_per_case=args.controls_per_case,
        random_state=args.random_state,
    )
    result.matched.to_parquet(args.output, index=False)
    result.audit.to_csv(args.audit, index=False)
    args.summary.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()
