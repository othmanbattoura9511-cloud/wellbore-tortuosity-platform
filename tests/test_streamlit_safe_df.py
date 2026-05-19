"""Streamlit-safe dataframe conversion."""

import pandas as pd
import pytest

from display_labels import make_streamlit_safe_dataframe


def test_none_returns_empty():
    assert make_streamlit_safe_dataframe(None).empty


def test_series_input():
    safe = make_streamlit_safe_dataframe(pd.Series([1, 2], name="MD"))
    assert len(safe.columns) == 1
    assert len(safe) == 2


def test_nested_object_becomes_string():
    df = pd.DataFrame({"MD": [1.0, 2.0], "meta": [{"a": 1}, {"b": 2}]})
    safe = make_streamlit_safe_dataframe(df)
    assert safe["MD"].dtype.kind in "fi"
    assert all(isinstance(v, str) for v in safe["meta"])


def test_multiindex_columns_flattened():
    df = pd.DataFrame({("DLS", "mean"): [1.0], ("DLS", "max"): [2.0]})
    safe = make_streamlit_safe_dataframe(df)
    assert len(safe.columns) == 2
    assert all(isinstance(c, str) for c in safe.columns)


def test_duplicate_column_names_deduped():
    df = pd.DataFrame([[1, 2]], columns=["MD", "MD"])
    safe = make_streamlit_safe_dataframe(df)
    assert len(safe.columns) == 2
    assert safe.columns.tolist() == ["MD", "MD_1"]


def test_non_dataframe_coerced():
    safe = make_streamlit_safe_dataframe({"x": [1, 2]})
    assert list(safe.columns) == ["x"]
