import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


LOOKBACK_DAYS = round(36 * 365.25 / 12)
INDEX_BUFFER_DAYS = round(12 * 365.25 / 12)
DEMOGRAPHIC_PREFIXES = ("AGE:", "GENDER:", "RACE:", "ETHNICITY:")
TOKEN_PATTERN = re.compile(r"'([^']*)'")


def as_tokens(value) -> list[str]:
    if isinstance(value, str):
        return TOKEN_PATTERN.findall(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [str(token) for token in value]


def as_positions(value) -> list[int]:
    if isinstance(value, str):
        return np.fromstring(value.strip()[1:-1], sep=" ", dtype=np.int64).tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [int(position) for position in value]


def get_baseline_age(tokens) -> int:
    for token in tokens:
        token = str(token)
        if token.startswith("AGE:"):
            return int(token.split(":", 1)[1])
    raise ValueError("Missing AGE token")


def rebuild_control(row: dict, full_record: pd.Series) -> dict:
    full_tokens = as_tokens(full_record["sorted_event_tokens"])
    full_positions = as_positions(full_record["day_position_tokens"])
    last_record_day = max(full_positions)
    index_day = max(0, last_record_day - INDEX_BUFFER_DAYS)
    lookback_start_day = max(0, index_day - LOOKBACK_DAYS)

    baseline_age = get_baseline_age(full_tokens)
    age_at_index = baseline_age + index_day / 365.25
    age_token = f"AGE:{round(age_at_index)}"

    output_tokens = []
    output_positions = []
    for token, position in zip(full_tokens, full_positions):
        is_demographic = token.startswith(DEMOGRAPHIC_PREFIXES) and position == 0
        if is_demographic:
            if token.startswith("AGE:"):
                token = age_token
            output_tokens.append(token)
            output_positions.append(0)
        elif lookback_start_day <= position <= index_day:
            output_tokens.append(token)
            output_positions.append(position)

    row.update(
        {
            "sorted_event_tokens": output_tokens,
            "day_position_tokens": output_positions,
            "index_day": index_day,
            "index_source": "12mo_before_last_record",
            "lookback_start_day": lookback_start_day,
            "lookback_months": 36,
            "baseline_age_years": float(baseline_age),
            "age_at_index_years": age_at_index,
            "estimated_index_year": float(row["YEAR_OF_BIRTH"]) + age_at_index,
            "original_event_count": len(full_tokens),
            "preindex_event_count": len(output_tokens),
            "removed_event_count": len(full_tokens) - len(output_tokens),
            "last_record_day": last_record_day,
        }
    )
    return row


def reindex_controls(full_record_path: Path, cohort_path: Path, output_path: Path) -> None:
    full_records = pd.read_parquet(full_record_path).set_index("person_id")
    cohort = pd.read_parquet(cohort_path)
    output_rows = []

    for row in cohort.to_dict(orient="records"):
        full_record = full_records.loc[row["person_id"]]
        if int(row["COHORT"]) == 0:
            row = rebuild_control(row, full_record)
        else:
            row["sorted_event_tokens"] = as_tokens(row["sorted_event_tokens"])
            row["day_position_tokens"] = as_positions(row["day_position_tokens"])
            row["last_record_day"] = max(as_positions(full_record["day_position_tokens"]))
        output_rows.append(row)

    output = pd.DataFrame(output_rows)
    table = pa.Table.from_pandas(output, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-index controls 12 months before their last record of any kind."
    )
    parser.add_argument("--full-records", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reindex_controls(args.full_records, args.cohort, args.output)


if __name__ == "__main__":
    main()
