# %%
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as feather


DATA_DIR = Path(r"D:\VibeCoding\kaggle\current_competitions\mscapital\data")
FILES = {
    "label": DATA_DIR / "train" / "label.feather",
    "test_transaction": DATA_DIR / "test" / "transaction.feather",
}

_tables = {}


def get_table(file_key):
    """Memory-map a Feather table and cache it for later pages."""
    if file_key not in FILES:
        raise KeyError(f"Unknown file key: {file_key}. Available: {list(FILES)}")
    if file_key not in _tables:
        _tables[file_key] = feather.read_table(FILES[file_key], memory_map=True)
    return _tables[file_key]


def describe(file_key):
    table = get_table(file_key)
    print(f"File: {FILES[file_key]}")
    print(f"Rows: {table.num_rows:,}")
    print(f"Columns: {table.num_columns}")
    print(table.schema)


def show_page(file_key, page=1, page_size=100, columns=None):
    """Return one page as a pandas DataFrame for VS Code's table viewer."""
    if page < 1:
        raise ValueError("page must be at least 1")
    if page_size < 1:
        raise ValueError("page_size must be at least 1")

    table = get_table(file_key)
    if columns:
        table = table.select(columns)

    start = (page - 1) * page_size
    if start >= table.num_rows:
        raise ValueError(f"Page starts after the final row ({table.num_rows:,})")
    return table.slice(start, page_size).to_pandas()


def show_sample(file_key, sample_id, max_rows=1000):
    """Return rows belonging to one sample_id."""
    table = get_table(file_key)
    if "sample_id" not in table.column_names:
        raise ValueError(f"{file_key} has no sample_id column")
    mask = pc.equal(table["sample_id"], sample_id)
    return table.filter(mask).slice(0, max_rows).to_pandas()


# %%
# Run this cell to inspect the complete label table in VS Code's Data Viewer.
describe("label")
label_df = get_table("label").to_pandas()
label_df


# %%
# Change PAGE and PAGE_SIZE, then run this cell to browse transaction rows.
PAGE = 1
PAGE_SIZE = 200
transaction_page_df = show_page(
    "test_transaction",
    page=PAGE,
    page_size=PAGE_SIZE,
)
transaction_page_df


# %%
# Change SAMPLE_ID to inspect all available transaction rows for one sample.
SAMPLE_ID = 0
MAX_SAMPLE_ROWS = 1000
transaction_sample_df = show_sample(
    "test_transaction",
    sample_id=SAMPLE_ID,
    max_rows=MAX_SAMPLE_ROWS,
)
transaction_sample_df
