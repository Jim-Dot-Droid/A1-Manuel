# Crash Predictor — Manual Bets with Next Round Signal

This is a Streamlit app for manually tracking SOL bets based on multiplier history.  
It allows you to:

- Upload a CSV file of multipliers or add them manually.
- Predict "Above" or "Under" based on the number of unders in the last 20 rounds.
- Place manual bets with full control over prediction and bet amount.
- Signal a bet for the **next round** before the multiplier is known.
- Track your manual balance and prediction accuracy.

## How to Use

1. Install dependencies (see `requirements.txt`).
2. Run the app:
   ```bash
   streamlit run app.py

