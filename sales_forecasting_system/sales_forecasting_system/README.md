# Sales Forecasting System

A local Streamlit application that forecasts sales from CSV/Excel history using a naive baseline, ARIMA, and exponential smoothing (ETS).

## Requirements
- Python 3.10+
- Windows, macOS, or Linux
- Internet connection for the first dependency installation

## Install and run (Windows PowerShell)
```powershell
cd sales_forecasting_system
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.synthetic_data
python -m pytest -q
streamlit run app/streamlit_app.py
```

If PowerShell blocks activation, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Install and run (macOS/Linux)
```bash
cd sales_forecasting_system
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.synthetic_data
python -m pytest -q
streamlit run app/streamlit_app.py
```

## Use
1. Open the local URL printed by Streamlit (usually `http://localhost:8501`).
2. Upload `.csv`, `.xlsx`, or `.xls`.
3. Review the preview and detected date/sales columns. Confirm them in the dropdowns.
4. Choose frequency, forecast horizon and model, then click **Generate forecast**.
5. Review historical/forecast chart, model metrics, quality report and warnings.
6. Download forecast and model metrics as CSV.

## Dataset
Place the provided `Global_Superstore2.csv` in the project root or upload it through the UI. The app does not hard-code a dataset path.

## Structure
```text
sales_forecasting_system/
├── app/
│   └── streamlit_app.py
├── data/
│   └── synthetic/
├── docs/
│   ├── contracts.md
│   └── developer_guide.md
├── src/
│   ├── analysis.py
│   ├── column_detection.py
│   ├── config.py
│   ├── contracts.py
│   ├── engine.py
│   ├── evaluation.py
│   ├── ingestion.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── serialization.py
│   └── synthetic_data.py
├── tests/
├── requirements.txt
└── README.md
```

## Design principles
- Core engine never imports Streamlit.
- Data exchanged between stages uses dataclasses.
- Warnings are structured and carried into results.
- Validation is chronological only; no random splits.
- Forecast intervals are approximate, not guarantees.
- Scope v1 is sales forecasting only; purchases, multi-series forecasting and REST API are future backlog items.

## Limitations
- Automatic column/date inference is heuristic. Confirm columns when warned.
- Forecast quality depends on history length and regularity.
- The default daily/weekly/monthly/quarterly frequency inference is a suggestion, not ground truth.
- ARIMA/ETS may be ineligible for short or constant series; the baseline remains available.
