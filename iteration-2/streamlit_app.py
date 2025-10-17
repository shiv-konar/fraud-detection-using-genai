import streamlit as st
import pandas as pd
import io
from streamlit_fraud_detector import (
    parse_timestamp,
    Transaction,
    score_transaction,
    severity_label
)
from typing import List


# --- Helper Functions for Streamlit Integration ---

# We need to adapt read_transactions to handle the file-like object from Streamlit
@st.cache_data
def read_transactions_from_uploaded_file(uploaded_file):
    """Reads transactions from a Streamlit uploaded file object."""

    string_data = uploaded_file.getvalue().decode("utf-8")

    txns: List[Transaction] = []

    reader = pd.read_csv(io.StringIO(string_data)).to_dict('records')

    for row in reader:
        txns.append(Transaction(
            txn_id=str(row["txn_id"]),
            user_id=str(row["user_id"]),
            amount=float(row["amount"]),
            currency=str(row["currency"]),
            card_country=str(row["card_country"]),
            ip_country=str(row["ip_country"]),
            merchant_mcc=int(row["merchant_mcc"]),
            device_id=str(row["device_id"]),
            # parse_timestamp is imported directly
            timestamp=parse_timestamp(str(row["timestamp"])),
        ))
    # Sort by timestamp to make velocity/device rules more sensible
    txns.sort(key=lambda t: t.timestamp)
    return txns


# --- Streamlit UI and Logic ---

st.set_page_config(
    page_title="Rule-Based Fraud Detection App",
    layout="wide",
)

st.title("💸 Rule-Based Fraud Detection App")
st.markdown("Upload a CSV file containing transactions to analyze for fraud scores.")

# --- Sidebar for Configuration ---
st.sidebar.header("Rule Configuration")

# Risk Threshold
flag_threshold = st.sidebar.slider(
    "Flagging Risk Score Threshold",
    min_value=10, max_value=100, value=50, step=5,
    help="Transactions with a score equal to or above this value will be flagged."
)
st.sidebar.markdown("---")

# High Amount Rule
st.sidebar.subheader("High Amount Rule")
high_amount_threshold = st.sidebar.number_input(
    "Amount Threshold (USD)",
    min_value=100.0, value=2000.0, step=100.0
)
st.sidebar.markdown("---")

# Velocity Rule
st.sidebar.subheader("High Velocity Rule")
velocity_window_minutes = st.sidebar.number_input(
    "Window (minutes)",
    min_value=1, value=5, step=1
)
velocity_max_allowed = st.sidebar.number_input(
    "Max Transactions Allowed in Window",
    min_value=1, value=3, step=1
)
st.sidebar.markdown("---")

# Merchant Blacklist (Simplified input for Streamlit)
st.sidebar.subheader("Merchant Blacklist Rule")
mcc_input = st.sidebar.text_input(
    "Blacklisted MCCs (comma separated)",
    "4829, 7995"
)
try:
    blacklisted_mcc = [int(mcc.strip()) for mcc in mcc_input.split(',') if mcc.strip()]
except ValueError:
    st.sidebar.error("Invalid MCC list. Must be integers.")
    blacklisted_mcc = []

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv"
)

if uploaded_file is not None:
    # --- Main Processing ---

    # 1. Build the Configuration Dictionary
    config = {
        "high_amount_threshold": high_amount_threshold,
        "velocity_window_minutes": velocity_window_minutes,
        "velocity_max_allowed": velocity_max_allowed,
        "blacklisted_mcc": blacklisted_mcc,
    }

    st.info(f"Processing transactions using Flag Threshold: **{flag_threshold}**")

    try:
        # Streamlit allows us to easily use a function that returns a DataFrame
        @st.cache_data(show_spinner="Analyzing transactions...")
        def analyze_transactions(uploaded_file, flag_threshold, config):
            # Using the helper function to get the list of Transaction objects
            txns = read_transactions_from_uploaded_file(uploaded_file)

            user_history = {}
            seen_devices = {}
            results = []

            for t in txns:
                score, rule_results = score_transaction(t, user_history, seen_devices, config)

                results.append({
                    "Timestamp": t.timestamp.isoformat(),
                    "Txn ID": t.txn_id,
                    "User ID": t.user_id,
                    "Amount": t.amount,
                    "Currency": t.currency,
                    "Score": score,
                    # FIX: Call severity_label directly, as it was imported directly.
                    "Severity": severity_label(score),
                    "Status": "FLAG" if score >= flag_threshold else "OK",
                    "Rules Triggered": "; ".join([f"{rr.name} (+{rr.points})" for rr in rule_results]),
                })

                # Update history/devices
                user_history.setdefault(t.user_id, []).append(t)
                seen_devices.setdefault(t.user_id, set()).add(t.device_id)

            return pd.DataFrame(results)


        # Run the analysis and get the results DataFrame
        results_df = analyze_transactions(uploaded_file, flag_threshold, config)

        # --- Display Results ---

        st.header("Detection Summary")

        flagged_count = results_df[results_df['Status'] == 'FLAG'].shape[0]
        ok_count = results_df.shape[0] - flagged_count

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", results_df.shape[0])
        col2.metric("Flagged Transactions", flagged_count)
        col3.metric("OK Transactions", ok_count)

        st.header("Detailed Transaction Results")


        # Function to apply styling to the DataFrame
        def style_fraud_df(df):
            # Color coding for severity
            def color_severity(val):
                color = ''
                if val == 'HIGH':
                    color = '#FFCCCC'  # Red
                elif val == 'MEDIUM':
                    color = '#FFE0B2'  # Orange
                elif val == 'LOW':
                    color = '#FFFFE0'  # Yellow
                elif val == 'INFO':
                    color = '#CCFFCC'  # Green
                return f'background-color: {color}'

            # Bold text for flagged status
            def highlight_status(val):
                if val == 'FLAG': return 'font-weight: bold; color: darkred;'
                return ''

            return df.style \
                .applymap(color_severity, subset=['Severity']) \
                .applymap(highlight_status, subset=['Status']) \
                .format({'Amount': '${:,.2f}'})


        st.dataframe(
            style_fraud_df(results_df),
            use_container_width=True,
            # Streamlit table configuration for a better look
            column_order=["Timestamp", "Txn ID", "User ID", "Amount", "Score", "Severity", "Status", "Rules Triggered"]
        )

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        st.exception(e)  # Display the full error trace for debugging

# --- Instructions for users without a file ---
else:
    st.info("Please upload a CSV file to begin the fraud detection analysis.")
    st.markdown("""
        **Expected CSV columns:**
        `txn_id`, `user_id`, `amount`, `currency`, `card_country`, 
        `ip_country`, `merchant_mcc`, `device_id`, `timestamp` (ISO 8601 format)
    """)