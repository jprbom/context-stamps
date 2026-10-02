"""Bounded public-input checks for generated record-union programs.

This is an explicit host-declared task contract, not a benchmark verifier or a
model-generated success claim. Run only in an isolated environment authorized
to read the declared inputs and generated outputs.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 100000


def bounded(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("missing or oversized declared file")
    return path


def keys(path, column):
    path = bounded(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if column not in (reader.fieldnames or []):
                raise ValueError("declared identity column absent")
            values = []
            for row in reader:
                if len(values) >= MAX_ROWS:
                    raise ValueError("too many records")
                values.append(row[column])
    elif path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not all(isinstance(row, dict) and column in row
                                                    for row in records):
            raise ValueError("declared JSON record identity absent")
        values = [row[column] for row in records]
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        if pq.ParquetFile(path).metadata.num_rows > MAX_ROWS:
            raise ValueError("too many records")
        table = pq.read_table(path, columns=[column])
        values = table[column].to_pylist()
    else:
        raise ValueError("unsupported declared format")
    if len(values) > MAX_ROWS:
        raise ValueError("too many records")
    try:
        result = [int(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError("noninteger record identity") from error
    if any(str(value).strip() != str(number) for value, number in zip(values, result)):
        raise ValueError("noncanonical record identity")
    return result


def check(sources, output, report, *, output_column="user_id", required_columns=()):
    if not 1 <= len(sources) <= 16:
        raise ValueError("one to 16 declared inputs required")
    expected = set()
    for path, column in sources:
        expected.update(keys(path, column))
    output = bounded(output)
    if output.suffix != ".parquet":
        raise ValueError("Parquet output required")
    import pyarrow.parquet as pq

    columns = set(pq.read_schema(output).names)
    observed = keys(output, output_column)
    report_data = json.loads(bounded(report).read_text(encoding="utf-8"))
    if not isinstance(report_data, dict) or not isinstance(report_data.get("conflicts"), list):
        raise ValueError("conflict report structure invalid")
    conflict_count = report_data.get("total_conflicts")
    if type(conflict_count) is not int or conflict_count < 0 or conflict_count > MAX_ROWS:
        raise ValueError("conflict count invalid")
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    duplicates = sorted(key for key, count in Counter(observed).items() if count > 1)
    absent_columns = sorted(set(required_columns) - columns)
    count_matches = conflict_count == len(report_data["conflicts"])
    return {"status": "complete" if not (missing or extra or duplicates or absent_columns)
            and count_matches else "incomplete",
            "expected_count": len(expected), "output_count": len(observed),
            "missing_ids": missing[:20], "unexpected_ids": extra[:20],
            "duplicate_ids": duplicates[:20], "missing_columns": absent_columns,
            "conflict_count_matches_list": count_matches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs=2, action="append", metavar=("PATH", "ID_COLUMN"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-column", default="user_id")
    parser.add_argument("--required-column", action="append", default=[])
    args = parser.parse_args()
    try:
        result = check(args.source, args.output, args.report, output_column=args.output_column,
                       required_columns=args.required_column)
    except (OSError, ValueError, KeyError, ImportError) as error:
        result = {"status": "unavailable", "reason": type(error).__name__}
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
