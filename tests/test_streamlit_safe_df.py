"""Streamlit-safe dataframe conversion."""

import pandas as pd

from display_labels import make_streamlit_safe_dataframe


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
