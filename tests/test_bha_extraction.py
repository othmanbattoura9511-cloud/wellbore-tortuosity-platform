from bha_extraction import (
    _extract_hole_size_from_bit_row,
    _extract_hole_size_from_filename,
    _extract_hole_size_from_pdf_text,
    extract_bha_from_text,
    normalize_bha_intervals,
)


def test_extract_motor_and_md():
    text = """
    BHA REPORT
    MD IN: 1000 M
    MD OUT: 2500 M
    COMPONENTS:
    Steerable Motor 6 3/4
    MWD
    HOLE SIZE: 8 1/2 IN
    BIT TYPE: PDC
    """
    row = extract_bha_from_text(text, source_file="run1.pdf", run_index=1)
    assert row["Drilling_System"] == "Motor"
    assert row["MD_In"] == 1000.0
    assert row["MD_Out"] == 2500.0
    assert "8" in row["Hole_Size"] or row["Hole_Size"] != "Unknown"


def test_extract_mud_motor():
    row = extract_bha_from_text("COMPONENT LIST\nMud Motor\nPDM 5.0\nMD IN 100 MD OUT 500")
    assert row["Drilling_System"] == "Motor"


def test_extract_rss_push():
    text = "RSS ASSEMBLY PUSH THE BIT STEERING MD IN 500 MD OUT 1200"
    row = extract_bha_from_text(text)
    assert row["Drilling_System"] == "RSS"
    assert row["RSS_Type"] == "Push-the-bit"


def test_extract_powerdrive_push():
    text = "BHA COMPONENTS:\nPowerDrive Xceed\nPush-the-bit\nMD IN 800 MD OUT 2000"
    row = extract_bha_from_text(text)
    assert row["Drilling_System"] == "RSS"
    assert row["RSS_Type"] == "Push-the-bit"


def test_extract_geopilot_point():
    text = "Tool string: GeoPilot X\nPoint-the-bit RSS\nMD IN 1200 MD OUT 3400"
    row = extract_bha_from_text(text)
    assert row["Drilling_System"] == "RSS"
    assert row["RSS_Type"] == "Point-the-bit"


def test_extract_autotrak_rotary_steerable():
    text = "AutoTrak rotary steerable assembly\nMD IN 400 MD OUT 900"
    row = extract_bha_from_text(text)
    assert row["Drilling_System"] == "RSS"


def test_steerable_motor_not_classified_as_rss():
    text = "Rotary steerable motor steerable motor section\nSteerable Motor 8 inch"
    row = extract_bha_from_text(text)
    assert row["Drilling_System"] == "Motor"


def test_hole_size_from_pdf_label():
    assert _extract_hole_size_from_pdf_text("HOLE SIZE: 8 1/2 IN") == "8 1/2 in"


def test_hole_size_from_bit_row_line():
    text = "3  PDC Bit  8 1/2 in  SN 12345"
    assert _extract_hole_size_from_bit_row(text) == "8 1/2 in"


def test_hole_size_from_bit_table():
    tables = [
        [
            ["Item", "Description", "Bit Size", "Length"],
            ["1", "MWD", "", "30"],
            ["2", "PDC Bit", "6.75", "8"],
        ]
    ]
    assert _extract_hole_size_from_bit_row("", tables) == "6.75 in"


def test_hole_size_from_filename():
    assert _extract_hole_size_from_filename("BHA_HoleSize_8.5_Run1.pdf") == "8.5 in"
    assert _extract_hole_size_from_filename("8_1_2_in_BHA.pdf") == "8 1/2 in"


def test_hole_size_filename_fallback_when_pdf_missing():
    row = extract_bha_from_text("MUD MOTOR BHA", source_file="BHA_6.75in_run.pdf")
    assert row["Hole_Size"] == "6.75 in"


def test_normalize_intervals():
    df = normalize_bha_intervals(
        __import__("pandas").DataFrame(
            [{"BHA_Run": "Run 1", "MD_In": None, "MD_Out": None, "Drilling_System": "Unknown"}]
        )
    )
    assert "BHA" in df.columns
    assert df["Drilling_System"].iloc[0] == "Unknown"
