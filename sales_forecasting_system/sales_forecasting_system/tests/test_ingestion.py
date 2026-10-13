from io import BytesIO
import pandas as pd
import pytest
from src.ingestion import load_file
from src.contracts import FileFormatError

def test_load_csv(tmp_path):
    p = tmp_path / "sales.csv"
    pd.DataFrame({"Order Date": ["2024-01-01", "2024-01-02"], "Sales": [10, 20]}).to_csv(p, index=False)
    result = load_file(p)
    assert len(result.frame) == 2
    assert result.file_info.format == "CSV"

def test_reject_extension(tmp_path):
    p = tmp_path / "sales.txt"
    p.write_text("date,sales\n2024-01-01,10")
    with pytest.raises(FileFormatError):
        load_file(p)

def test_semicolon_csv(tmp_path):
    p = tmp_path / "sales.csv"
    p.write_text("date;sales\n2024-01-01;10\n2024-01-02;20", encoding="utf-8")
    result = load_file(p)
    assert result.frame.shape == (2, 2)
