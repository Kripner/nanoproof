#!/usr/bin/env python3
import os
import sys
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


def inspect_parquet(path: str) -> None:
    """Inspect Parquet file structure and compute stats."""
    pf = pq.ParquetFile(path)
    metadata = pf.metadata
    # Use the Arrow schema for column inspection
    schema: pa.Schema = pf.schema_arrow

    print("=== Parquet File Overview ===")
    print(f"File path: {path}")
    print(f"Num row groups: {pf.num_row_groups}")
    print(f"Num columns: {len(schema)}")
    print(f"Total rows (samples): {metadata.num_rows}\n")

    print("=== Schema ===")
    for field in schema:
        print(f"  - {field.name}: {field.type}")
    print()

    print("=== Row Groups ===")
    for rg_idx in range(pf.num_row_groups):
        rg_meta = metadata.row_group(rg_idx)
        rg_rows = rg_meta.num_rows
        rg_total_bytes = sum(
            rg_meta.column(col_idx).total_compressed_size
            for col_idx in range(rg_meta.num_columns)
        )
        print(
            f"  Row group {rg_idx}: "
            f"rows={rg_rows}, compressed_size≈{rg_total_bytes} bytes"
        )
    print()

    # Compute total number of characters in `text` column
    print("=== Text Column Statistics ===")
    if "text" not in schema.names:
        print("Column 'text' not found in schema.")
        return

    text_idx = schema.get_field_index("text")
    text_field = schema.field(text_idx)

    if not (pa.types.is_string(text_field.type) or pa.types.is_large_string(text_field.type)):
        print(
            f"Warning: 'text' column is type {text_field.type}, not string; "
            "attempting to cast when reading."
        )
        cast_to_string = True
    else:
        cast_to_string = False

    total_chars = 0
    total_non_null_rows = 0
    batch_size = 65536

    for batch in pf.iter_batches(columns=["text"], batch_size=batch_size):
        arr = batch.column(0)

        if cast_to_string:
            arr = pc.cast(arr, pa.string())

        # Length of each string; nulls become null
        lengths = pc.utf8_length(arr)
        # Sum of lengths (ignores nulls)
        batch_char_sum = pc.sum(lengths).as_py() or 0
        total_chars += batch_char_sum

        # Count non-null entries
        non_null_count = batch.num_rows - arr.null_count
        total_non_null_rows += non_null_count

    print(f"Total non-null rows in 'text' column: {total_non_null_rows}")
    print(f"Total number of characters in 'text' column: {total_chars}")
    print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_parquet.py <parquet_file>", file=sys.stderr)
        sys.exit(1)
    
    dest = sys.argv[1]

    if not os.path.exists(dest):
        print(f"Error: File {dest} does not exist", file=sys.stderr)
        sys.exit(1)

    inspect_parquet(dest)


if __name__ == "__main__":
    main()
