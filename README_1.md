# COVID-19 Many-to-Many Simple RNN Forecaster

Predicts the **next 7 days** of confirmed COVID-19 cases from the **previous 7 days**,
using a many-to-many Simple RNN (no LSTM/GRU).

## Files
- `day_wise.csv` — sample day-wise COVID dataset (Date, Confirmed, Deaths, Recovered, Active, etc.)
- `train.py` — builds sequences, trains the Simple RNN, saves `covid_rnn_model.keras` + `scaler.pkl`
- `app.py` — Streamlit app: pick a 7-day window, predict the next 7 days, plot actual vs predicted
- `requirements.txt` — dependencies
- `covid_rnn_model.keras` / `scaler.pkl` — already-trained model included so you can run the app immediately

## Setup
```bash
pip install -r requirements.txt
```

## 1. Train (optional — a trained model is already included)
```bash
python train.py
```
This creates `covid_rnn_model.keras` and `scaler.pkl`.

## 2. Run the app
```bash
streamlit run app.py
```
Use the sidebar slider to pick which 7-day window to feed in, then click
**"Predict next 7 days"** to see the forecast table and the actual-vs-predicted graph.

## Model architecture
```
Input (7, 1)
 → SimpleRNN(64, return_sequences=True)
 → SimpleRNN(32)
 → Dense(32, relu)
 → Dense(7)          # 7 outputs at once = many-to-many
```

## Customizing
- To forecast a different column, change `FEATURE_COL` in `train.py` (e.g. `"Active"`)
  and retrain — `app.py` will automatically pick up the new config from `scaler.pkl`.
- To forecast multiple features at once (Confirmed, Deaths, Recovered, Active together),
  extend `make_sequences` in `train.py` to keep all 4 columns and change the final
  `Dense` layer to `Dense(n_steps_out * n_features)` reshaped to `(n_steps_out, n_features)`.

## Note on the dataset
`day_wise.csv` included here is a **synthetically generated** dataset shaped like the
real Kaggle "COVID-19 day wise" dataset (same columns, realistic logistic-growth curve).
If you have the real dataset, just replace `day_wise.csv` with it — the code works as-is
since it only relies on the `Date` and `Confirmed` (or whichever feature you choose) columns.
