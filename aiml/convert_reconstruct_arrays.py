import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TOKEN_PATTERN = re.compile(r"'([^']*)'")


def parse_token_array(value: str) -> list[str]:
    return TOKEN_PATTERN.findall(value)


def parse_position_array(value: str) -> list[int]:
    return np.fromstring(value.strip()[1:-1], sep=" ", dtype=np.int64).tolist()


def convert(input_path: Path, output_path: Path) -> None:
    table = pq.read_table(input_path)
    tokens = [parse_token_array(value) for value in table["sorted_event_tokens"].to_pylist()]
    positions = [parse_position_array(value) for value in table["day_position_tokens"].to_pylist()]

    mismatches = [
        index
        for index, (token_row, position_row) in enumerate(zip(tokens, positions))
        if len(token_row) != len(position_row)
    ]
    if mismatches:
        raise ValueError(f"Token/position length mismatch in rows: {mismatches[:10]}")

    token_index = table.schema.get_field_index("sorted_event_tokens")
    position_index = table.schema.get_field_index("day_position_tokens")
    table = table.set_column(
        token_index,
        "sorted_event_tokens",
        pa.array(tokens, type=pa.list_(pa.string())),
    )
    table = table.set_column(
        position_index,
        "day_position_tokens",
        pa.array(positions, type=pa.list_(pa.int64())),
    )
    pq.write_table(table, output_path, compression="snappy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert serialized sequence columns to Parquet list columns.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
