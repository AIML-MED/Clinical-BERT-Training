import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PREFIX_MAPPING = {
    "ICD-10-CM:": "ICD10CM:",
    "SNOMED-CT:": "SNOMED:",
    "RxNorm_drug:": "RXNORM:",
}

TOKEN_MAPPING = {
    "RACE:white": "RACE:White",
    "RACE:black": "RACE:Black",
    "RACE:asian": "RACE:Asian",
    "RACE:other": "RACE:Other",
    "ETHNICITY:hispanic": "ETHNICITY:Hispanic",
    "ETHNICITY:unknown": "ETHNICITY:Unknown",
}


def normalize_token(token: str) -> str:
    if token in TOKEN_MAPPING:
        return TOKEN_MAPPING[token]
    for source, target in PREFIX_MAPPING.items():
        if token.startswith(source):
            return target + token[len(source) :]
    return token


def normalize_parquet(input_path: Path, output_path: Path) -> None:
    table = pq.read_table(input_path)
    field = table.schema.field("sorted_event_tokens")
    if not pa.types.is_list(field.type) and not pa.types.is_large_list(field.type):
        raise TypeError("sorted_event_tokens must be a Parquet list column")

    normalized = [
        [normalize_token(token) for token in row]
        for row in table["sorted_event_tokens"].to_pylist()
    ]
    column_index = table.schema.get_field_index("sorted_event_tokens")
    table = table.set_column(
        column_index,
        "sorted_event_tokens",
        pa.array(normalized, type=pa.list_(pa.string())),
    )
    pq.write_table(table, output_path, compression="snappy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Synthea tokens to OSCAR naming conventions.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    normalize_parquet(args.input, args.output)


if __name__ == "__main__":
    main()
