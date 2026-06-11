import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def rebase_positions(input_path: Path, output_path: Path) -> None:
    table = pq.read_table(input_path)
    positions = table["day_position_tokens"].to_pylist()
    starts = table["lookback_start_day"].to_pylist()

    rebased = []
    for row, start in zip(positions, starts):
        rebased.append([0 if position == 0 else position - start + 1 for position in row])

    if any(position < 0 for row in rebased for position in row):
        raise ValueError("Rebased positions must be non-negative")

    column_index = table.schema.get_field_index("day_position_tokens")
    table = table.set_column(
        column_index,
        "day_position_tokens",
        pa.array(rebased, type=pa.list_(pa.int64())),
    )
    pq.write_table(table, output_path, compression="snappy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebase event positions to the lookback window.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rebase_positions(args.input, args.output)


if __name__ == "__main__":
    main()
