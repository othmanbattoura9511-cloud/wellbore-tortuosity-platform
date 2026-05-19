from bha_extraction import extract_bha_from_text, normalize_bha_intervals


def test_extract_motor_and_md():
    text = """
    BHA REPORT
    MD IN: 1000 M
    MD OUT: 2500 M
    MUD MOTOR 6 3/4
    HOLE SIZE: 8 1/2 IN
    BIT TYPE: PDC
    """
    row = extract_bha_from_text(text, source_file="run1.pdf", run_index=1)
    assert row["Drilling_System"] == "Motor"
    assert row["MD_In"] == 1000.0
    assert row["MD_Out"] == 2500.0
    assert "8" in row["Hole_Size"] or row["Hole_Size"] != "Unknown"


def test_extract_rss_push():
    text = "RSS ASSEMBLY PUSH THE BIT STEERING MD IN 500 MD OUT 1200"
    row = extract_bha_from_text(text)
    assert row["Drilling_System"] == "RSS"
    assert row["RSS_Type"] == "Push-the-bit"


def test_normalize_intervals():
    df = normalize_bha_intervals(
        __import__("pandas").DataFrame(
            [{"BHA_Run": "Run 1", "MD_In": None, "MD_Out": None, "Drilling_System": "Unknown"}]
        )
    )
    assert "BHA" in df.columns
    assert df["Drilling_System"].iloc[0] == "Unknown"
