import pandas as pd
from src.column_detection import detect_columns, apply_user_selection

def test_detect_clear_columns():
    df = pd.DataFrame({"Order Date": pd.date_range("2024-01-01", periods=30), "Sales": range(30), "Customer ID": range(30)})
    result = detect_columns(df)
    assert result.selected_date == "Order Date"
    assert result.selected_sales == "Sales"

def test_override_columns():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "amount": [1,2,3,4,5]})
    result = detect_columns(df)
    result = apply_user_selection(result, df, "date", "amount")
    assert result.selected_sales == "amount"
    assert not result.needs_confirmation
