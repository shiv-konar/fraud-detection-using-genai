# streamlit_app.py
import streamlit as st
import pandas as pd
import io
import os
from typing import List, Dict
from streamlit_fraud_detector import (
    parse_timestamp,
    Transaction,
    score_transaction,
    severity_label
)
from llm_fraud_assistant import explain_fraud_decision, summarize_flagged_transactions

st.set_page_config(page_title="Rule-Based Fraud Detection + LLM", layout="wide")

st.title("💸 Rule-Based Fraud Detection App — with LLM Explanations")

st.sidebar.header("Rule Configuration")

flag_threshold = st.sidebar.slider(
    "Flagging Risk Score Threshold",
    min_value=10, max_value=100, value=50, step=5
)

st.sidebar.markdown("---")
st.sidebar.subheader("High Amount Rule")
high_amount_threshold = st.sidebar.number_input(
    "Amount Threshold (USD)",
    min_value=0.0, value=2000.0, step=100.0
)

st.sidebar.markdown("---")
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
st.sidebar.subheader("Merchant Blacklist")
mcc_input = st.sidebar.text_input("Blacklisted MCCs (comma separated)", "4829, 7995")
try:
    blacklisted_mcc = [int(mcc.strip()) for mcc in mcc_input.split(',') if mcc.strip()]
except ValueError:
    st.sidebar.error("Invalid MCC list. Must be integers.")
    blacklisted_mcc = []

st.sidebar.markdown("---")
st.sidebar.header("LLM / OpenAI settings")
st.sidebar.write("The app will read the OpenAI API key from `st.secrets['OPENAI_API_KEY']` when deployed,\n"
                 "or from the environment variable `OPENAI_API_KEY` when run locally (recommended).")

use_ai = st.sidebar.checkbox("Enable AI explanations", value=False)
show_summary = st.sidebar.checkbox("Show LLM summary of flagged transactions", value=False)

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

@st.cache_data
def read_transactions_from_uploaded_file(uploaded_file) -> List[Transaction]:
    string_data = uploaded_file.getvalue().decode("utf-8")
    df = pd.read_csv(io.StringIO(string_data))
    txns: List[Transaction] = []
    for _, row in df.iterrows():
        txns.append(Transaction(
            txn_id=str(row["txn_id"]),
            user_id=str(row["user_id"]),
            amount=float(row["amount"]),
            currency=str(row["currency"]),
            card_country=str(row["card_country"]),
            ip_country=str(row["ip_country"]),
            merchant_mcc=int(row["merchant_mcc"]),
            device_id=str(row["device_id"]),
            timestamp=parse_timestamp(str(row["timestamp"])),
        ))
    txns.sort(key=lambda t: t.timestamp)
    return txns

if uploaded_file is not None:
    config = {
        "high_amount_threshold": high_amount_threshold,
        "velocity_window_minutes": velocity_window_minutes,
        "velocity_max_allowed": velocity_max_allowed,
        "blacklisted_mcc": blacklisted_mcc
    }

    st.info(f"Processing transactions with flag threshold {flag_threshold}")

    @st.cache_data(show_spinner="Analyzing transactions...")
    def analyze_transactions(uploaded_file, flag_threshold, config):
        txns = read_transactions_from_uploaded_file(uploaded_file)
        user_history = {}
        seen_devices = {}
        flat_results = []
        for t in txns:
            score, rule_results = score_transaction(t, user_history, seen_devices, config)
            severity = severity_label(score)
            status = "FLAG" if score >= flag_threshold else "OK"
            flat_results.append({
                "txn_id": t.txn_id,
                "user_id": t.user_id,
                "amount": t.amount,
                "currency": t.currency,
                "card_country": t.card_country,
                "ip_country": t.ip_country,
                "merchant_mcc": t.merchant_mcc,
                "device_id": t.device_id,
                "timestamp": t.timestamp.isoformat(),
                "score": score,
                "severity": severity,
                "status": status,
                "rules_triggered": [{"name": r.name, "points": r.points, "details": r.details} for r in rule_results]
            })
            user_history.setdefault(t.user_id, []).append(t)
            seen_devices.setdefault(t.user_id, set()).add(t.device_id)
        return pd.DataFrame(flat_results), flat_results

    results_df, flat_results = analyze_transactions(uploaded_file, flag_threshold, config)

    st.header("Detection Summary")
    flagged_count = sum(1 for r in flat_results if r["status"] == "FLAG")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", results_df.shape[0])
    col2.metric("Flagged Transactions", flagged_count)
    col3.metric("OK Transactions", results_df.shape[0] - flagged_count)

    # Optionally generate AI explanations for flagged rows
    if use_ai:
        st.info("Generating AI explanations — this will call the OpenAI API.")
        # No secrets handling here; llm_fraud_assistant uses the OpenAI client which reads env or platform secrets.
        ai_explanations = []
        for r in flat_results:
            if r["status"] == "FLAG":
                txn_small = {
                    "txn_id": r["txn_id"],
                    "user_id": r["user_id"],
                    "amount": r["amount"],
                    "currency": r["currency"],
                    "card_country": r.get("card_country"),
                    "ip_country": r.get("ip_country"),
                    "timestamp": r.get("timestamp")
                }
                explanation = explain_fraud_decision(txn_small, r["rules_triggered"], r["severity"], r["score"])
                ai_explanations.append(explanation)
            else:
                ai_explanations.append("")
        results_df["ai_explanation"] = ai_explanations

    # Optionally ask LLM for a short summary across flagged transactions
    if show_summary:
        st.subheader("LLM Summary of Flagged Transactions")
        summary = summarize_flagged_transactions(flat_results, top_n=10)
        if summary:
            st.write(summary)
        else:
            st.write("Summary unavailable.")

    # Display table with styling
    def style_fraud_df(df):
        def color_severity(val):
            if val == 'HIGH': return 'background-color: #FFCCCC'
            if val == 'MEDIUM': return 'background-color: #FFE0B2'
            if val == 'LOW': return 'background-color: #FFFFE0'
            if val == 'INFO': return 'background-color: #CCFFCC'
            return ''
        def highlight_status(val):
            if val == 'FLAG': return 'font-weight: bold; color: darkred;'
            return ''
        styled = df.style \
            .applymap(color_severity, subset=['severity']) \
            .applymap(highlight_status, subset=['status']) \
            .format({'amount': '${:,.2f}'})
        return styled

    # Make sure the columns are present
    display_cols = ["timestamp", "txn_id", "user_id", "amount", "score", "severity", "status", "rules_triggered"]
    if "ai_explanation" in results_df.columns:
        display_cols.append("ai_explanation")

    st.header("Detailed Transaction Results")
    st.dataframe(style_fraud_df(results_df[display_cols]), use_container_width=True)

else:
    st.info("Please upload a CSV file with columns: txn_id, user_id, amount, currency, card_country, ip_country, merchant_mcc, device_id, timestamp")
