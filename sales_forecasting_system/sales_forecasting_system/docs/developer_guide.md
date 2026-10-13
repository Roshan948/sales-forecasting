# Developer guide

## Architecture
`Input -> ingestion -> column detection -> preprocessing/frequency detection -> model fitting -> chronological evaluation -> engine orchestration -> Streamlit UI`

The engine and all forecasting modules live under `src/`. The UI in `app/streamlit_app.py` calls `src.engine.run_forecast` and contains no model-fitting logic.

## Add a model
1. Implement the `Forecaster` interface in `src/models.py`.
2. Return a `ModelResult` with `eligible=False` and a reason when data is unsuitable.
3. Register the model in `build_forecaster`.
4. Add it to evaluation ordering and tests.
5. Ensure forecasts have exactly the requested horizon and chronological indices.

## Validation requirements
Never use random splits. Training indices must precede validation indices. Fit all model parameters using training data only. Keep intervals labeled approximate.
