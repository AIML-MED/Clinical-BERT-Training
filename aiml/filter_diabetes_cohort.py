import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DIABETES_FAMILIES = ("E08", "E09", "E10", "E11")


def get_diabetes_family(token: str) -> str | None:
    for prefix in ("ICD-10-CM:", "ICD10CM:"):
        if token.startswith(prefix):
            code = token[len(prefix) :].upper().replace(".", "")
            for family in DIABETES_FAMILIES:
                if code.startswith(family):
                    return family
    return None


def get_diabetes_events(tokens, positions) -> list[tuple[str, int]]:
    events = []
    for token, position in zip(tokens, positions):
        family = get_diabetes_family(str(token))
        if family is not None:
            events.append((family, int(position)))
    return events


def build_audit(full_record_path: Path, cohort_path: Path) -> pd.DataFrame:
    full_records = pd.read_parquet(
        full_record_path,
        columns=["person_id", "sorted_event_tokens", "day_position_tokens"],
    ).set_index("person_id")
    cohort = pd.read_parquet(cohort_path, columns=["person_id", "COHORT"])

    rows = []
    for record in cohort.itertuples(index=False):
        full_record = full_records.loc[record.person_id]
        events = get_diabetes_events(
            full_record["sorted_event_tokens"],
            full_record["day_position_tokens"],
        )
        first_e10_e11 = min(
            (day for family, day in events if family in ("E10", "E11")),
            default=None,
        )

        prior_e08_e09 = [
            (family, day)
            for family, day in events
            if family in ("E08", "E09")
            and first_e10_e11 is not None
            and day < first_e10_e11
        ]
        control_e08_e10 = [
            (family, day)
            for family, day in events
            if family in ("E08", "E09", "E10")
        ]

        exclude = False
        reason = ""
        if int(record.COHORT) == 1 and prior_e08_e09:
            exclude = True
            reason = "case_e08_e09_before_first_e10_e11"
        elif int(record.COHORT) == 0 and control_e08_e10:
            exclude = True
            reason = "control_any_e08_e09_e10"

        rows.append(
            {
                "person_id": record.person_id,
                "COHORT": int(record.COHORT),
                "exclude": exclude,
                "reason": reason,
                "first_e10_e11_day": first_e10_e11,
                "first_prior_e08_e09_day": min(
                    (day for _, day in prior_e08_e09),
                    default=None,
                ),
                "has_any_e08": any(family == "E08" for family, _ in events),
                "has_any_e09": any(family == "E09" for family, _ in events),
                "has_any_e10": any(family == "E10" for family, _ in events),
                "has_any_e11": any(family == "E11" for family, _ in events),
            }
        )

    return pd.DataFrame(rows)


def filter_cohort(cohort_path: Path, output_path: Path, audit: pd.DataFrame) -> None:
    table = pq.read_table(cohort_path)
    excluded_ids = set(audit.loc[audit["exclude"], "person_id"])
    keep = [person_id not in excluded_ids for person_id in table["person_id"].to_pylist()]
    pq.write_table(table.filter(pa.array(keep)), output_path, compression="snappy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply diabetes cohort leakage exclusions.")
    parser.add_argument("--full-records", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    audit = build_audit(args.full_records, args.cohort)
    filter_cohort(args.cohort, args.output, audit)
    audit.to_csv(args.audit, index=False)

    excluded = audit[audit["exclude"]]
    print(f"Input rows: {len(audit)}")
    print(f"Excluded rows: {len(excluded)}")
    print(f"Output rows: {len(audit) - len(excluded)}")
    print(excluded["reason"].value_counts().to_string())
    print(
        "Cases without E10/E11: "
        f"{len(audit[(audit['COHORT'] == 1) & audit['first_e10_e11_day'].isna()])}"
    )


if __name__ == "__main__":
    main()
