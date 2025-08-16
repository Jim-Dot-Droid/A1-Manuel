import streamlit as st
import pandas as pd
import numpy as np
import os

# File paths
HISTORY_FILE = "history.csv"
RESULTS_FILE = "results.csv"
MANUAL_FILE = "manual_balance.txt"

# Constants
INITIAL_BALANCE = 0.1
WINDOW = 20
MIN_UNDERS_FOR_ABOVE = 14

# --- Data helpers ---
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    if 'multiplier' in df.columns:
        return df['multiplier'].tolist()
    return df.iloc[:, 0].tolist()

def load_history():
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        return df['multiplier'].tolist()
    return []

def save_history(data):
    pd.DataFrame({'multiplier': data}).to_csv(HISTORY_FILE, index=False)

def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(columns=['prediction', 'actual', 'correct'])

def save_result(prediction, actual):
    correct = ((prediction == "Above") and actual > 2.0) or ((prediction == "Under") and actual <= 2.0)
    result_df = load_results()
    result_df.loc[len(result_df)] = [prediction, actual, correct]
    result_df.to_csv(RESULTS_FILE, index=False)

# --- Manual balance handlers ---
def get_manual_balance():
    if os.path.exists(MANUAL_FILE):
        with open(MANUAL_FILE, "r") as f:
            return float(f.read())
    return INITIAL_BALANCE

def update_manual_balance(prediction, actual, bet):
    balance = get_manual_balance()
    if (prediction == "Above" and actual > 2.0) or (prediction == "Under" and actual <= 2.0):
        balance += bet
    else:
        balance -= bet
    with open(MANUAL_FILE, "w") as f:
        f.write(str(balance))

def reset_balance():
    for f in [MANUAL_FILE, RESULTS_FILE]:
        if os.path.exists(f):
            os.remove(f)

# --- Logic ---
def normalize_input(value):
    return value / 100 if value > 10 else value

def predict_from_unders(data, threshold=2.0, window=WINDOW, min_unders_for_above=MIN_UNDERS_FOR_ABOVE):
    if len(data) < window:
        return None, None
    recent = np.array(data[-window:])
    under_count = int(np.sum(recent < threshold))
    return ("Above" if under_count >= min_unders_for_above else "Under", under_count)

# --- Streamlit App ---
def main():
    st.title("Crash Predictor — Manual Bets with Next Round Signal")

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = load_history()
    if "next_round_bet" not in st.session_state:
        st.session_state.next_round_bet = None  # tuple: (prediction, bet_amount)

    st.sidebar.header("Settings")
    min_unders = st.sidebar.slider("Min unders in last 20 to trigger 'Above' prediction",
                                   min_value=10, max_value=20, value=MIN_UNDERS_FOR_ABOVE, step=1)

    col1, col2 = st.columns([2,1])
    with col1:
        uploaded = st.file_uploader("Upload multipliers CSV", type=["csv"])
        if uploaded:
            st.session_state.history = load_csv(uploaded)
            save_history(st.session_state.history)
            st.success(f"Loaded {len(st.session_state.history)} multipliers.")

    with col2:
        if st.button("Reset all (balances & results)"):
            st.session_state.history = []
            save_history([])
            reset_balance()
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("Reset done.")

    # --- Next Round Bet Signal ---
    st.subheader("🎲 Signal Next Round Bet")
    next_pred = st.selectbox("Prediction for next round", ["Above", "Under"], key="next_pred")
    next_bet = st.number_input("Bet amount for next round (SOL)", min_value=0.0, value=0.01, step=0.01, key="next_bet_amount")

    if st.button("Signal Next Round Bet"):
        st.session_state.next_round_bet = (next_pred, next_bet)
        st.success(f"Next round bet signaled: {next_pred} with {next_bet} SOL")

    # --- Manual input ---
    st.subheader("Manual input")
    new_val = st.text_input("Enter multiplier (e.g., 1.87 or 187)")
    if st.button("Add multiplier"):
        try:
            val = float(new_val)
            val = normalize_input(val)

            # Apply next round bet if signaled
            if st.session_state.next_round_bet:
                pred, bet_amt = st.session_state.next_round_bet
                update_manual_balance(pred, val, bet_amt)
                st.success(f"Next round bet applied: {pred} with {bet_amt} SOL (multiplier = {val})")
                st.session_state.next_round_bet = None

            # Save result from prediction logic if exists
            if "last_prediction" in st.session_state:
                save_result(st.session_state.last_prediction, val)
                del st.session_state.last_prediction

            st.session_state.history.append(val)
            save_history(st.session_state.history)
            st.success(f"Added {val}x")

        except Exception as e:
            st.error("Invalid input")

    if st.session_state.history:
        data = st.session_state.history
        st.write(f"History length: {len(data)}")

        st.subheader("Prediction from Under Count (last 20)")
        prediction, under_count = predict_from_unders(data, min_unders_for_above=min_unders)
        if prediction:
            st.session_state.last_prediction = prediction
            st.write(f"Prediction: **{prediction}** (Under count in last 20 = {under_count})")
        else:
            st.write(f"Not enough data yet (need at least {WINDOW} rounds).")
    else:
        st.write("No history yet. Upload CSV or add multipliers manually.")

    st.subheader("Accuracy Tracker")
    results_df = load_results()
    if not results_df.empty:
        total = len(results_df)
        correct = int(results_df['correct'].sum())
        acc = correct / total if total else 0
        st.metric("Total Predictions", total)
        st.metric("Correct Predictions", correct)
        st.metric("Accuracy Rate", f"{acc:.1%}")
        st.dataframe(results_df[::-1].reset_index(drop=True))
    else:
        st.write("No verified predictions yet.")

    st.subheader("🎲 Manual Balance")
    st.metric("Manual Balance", f"{get_manual_balance():.4f} SOL")

if __name__ == "__main__":
    main()
