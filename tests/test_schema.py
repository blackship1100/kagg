from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.feather as feather

from mscapital.data.catalog import MappedFeatherFile
from mscapital.data.schema import compare_schema


class SchemaTests(unittest.TestCase):
    def test_equal_schemas_pass(self) -> None:
        schema = pa.schema([pa.field("sample_id", pa.int32())])
        self.assertTrue(compare_schema(schema, schema).ok)

    def test_type_and_extra_columns_are_reported(self) -> None:
        expected = pa.schema([pa.field("sample_id", pa.int32())])
        actual = pa.schema(
            [
                pa.field("sample_id", pa.int64()),
                pa.field("unexpected", pa.float32()),
            ]
        )
        result = compare_schema(actual, expected)
        self.assertFalse(result.ok)
        self.assertEqual(result.extra_columns, ("unexpected",))
        self.assertEqual(len(result.type_mismatches), 1)

    def test_mapped_feather_reuses_an_open_file_for_column_projection(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tiny.feather"
            feather.write_feather(
                pa.table(
                    {
                        "sample_id": pa.array([0, 0, 1], type=pa.int32()),
                        "value": pa.array([1.0, 2.0, 3.0], type=pa.float32()),
                    }
                ),
                path,
            )
            mapped = MappedFeatherFile(path)
            with mapped:
                self.assertEqual(mapped.schema.names, ["sample_id", "value"])
                projected = mapped.read_columns(["value"])
                self.assertEqual(projected.column_names, ["value"])
                self.assertEqual(projected.num_rows, 3)
            with self.assertRaises(RuntimeError):
                mapped.read_columns(["value"])


if __name__ == "__main__":
    unittest.main()
