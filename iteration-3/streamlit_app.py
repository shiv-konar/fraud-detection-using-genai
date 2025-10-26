# streamlit_app.py
import streamlit as st
import pandas as pd
import io
from typing import List, Dict
from streamlit_fraud_detector import (
    parse_timestamp,
    Transaction,
    score_transaction,
    severity_label
)
from llm_fraud_assistant import explain_fraud_decision, summarize_flagged_transactions


# --- Streamlit Page Config ---
st.set_page_config(page_title="Rule-Based Fraud Detection + LLM", layout="wide")
st.title("💸 Rule-Based Fraud Detection App — with LLM Explanations")

# --- Sidebar Configuration ---
st.sidebar.header("Rule Configuration")

flag_threshold = st.sidebar.slider(
    "Flagging Risk Score Threshold",
    min_value=10, max_value=100, value=50, step=5,
    help="Transactions scoring above this threshold will be flagged."
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
st.sidebar.header("AI Settings")
use_ai = st.sidebar.checkbox("Enable AI explanations", value=False)
show_summary = st.sidebar.checkbox("Show LLM summary of flagged transactions", value=False)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")


# --- Helper Functions ---
@st.cache_data
def read_transactions_from_uploaded_file(uploaded_file) -> List[Transaction]:
    """Parse uploaded CSV into a list of Transaction objects."""
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


# --- Main Processing ---
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
        user_history: Dict[str, List[Transaction]] = {}
        seen_devices: Dict[str, set] = {}
        results = []

        for t in txns:
            score, rule_results = score_transaction(t, user_history, seen_devices, config)
            severity = severity_label(score)
            status = "FLAG" if score >= flag_threshold else "OK"

            # Convert rule results into readable strings for display
            readable_rules = "; ".join(
                [f"{r.name} (+{r.points}) [{r.details}]" for r in rule_results]
            )

            results.append({
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
                "rules_triggered": readable_rules,  # ✅ FIX: make it a readable string
                "_rule_objects": [r.__dict__ for r in rule_results]  # hidden for AI use
            })

            user_history.setdefault(t.user_id, []).append(t)
            seen_devices.setdefault(t.user_id, set()).add(t.device_id)

        return pd.DataFrame(results), results

    results_df, flat_results = analyze_transactions(uploaded_file, flag_threshold, config)

    # --- Summary Metrics ---
    st.header("Detection Summary")
    flagged_count = sum(1 for r in flat_results if r["status"] == "FLAG")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(flat_results))
    col2.metric("Flagged Transactions", flagged_count)
    col3.metric("OK Transactions", len(flat_results) - flagged_count)

    # --- LLM Explanations ---
    if use_ai:
        st.info("Generating AI explanations for flagged transactions...")
        ai_explanations = []
        for r in flat_results:
            if r["status"] == "FLAG":
                txn_small = {
                    "txn_id": r["txn_id"],
                    "user_id": r["user_id"],
                    "amount": r["amount"],
                    "currency": r["currency"],
                    "card_country": r["card_country"],
                    "ip_country": r["ip_country"],
                    "timestamp": r["timestamp"]
                }
                explanation = explain_fraud_decision(
                    txn_small,
                    r["_rule_objects"],
                    r["severity"],
                    r["score"]
                )
                ai_explanations.append(explanation)
            else:
                ai_explanations.append("")
        results_df["ai_explanation"] = ai_explanations

    # --- Optional LLM Summary ---
    if show_summary:
        st.subheader("🧠 LLM Summary of Flagged Transactions")
        summary = summarize_flagged_transactions(flat_results, top_n=10)
        if summary:
            st.markdown(summary)
        else:
            st.warning("No summary available.")

    # --- Display Table with Styling ---
    def style_fraud_df(df):
        def color_severity(val):
            colors = {
                "HIGH": "#FFCCCC",
                "MEDIUM": "#FFE0B2",
                "LOW": "#FFFFE0",
                "INFO": "#CCFFCC"
            }
            return f'background-color: {colors.get(val, "")}'
        def highlight_status(val):
            return 'font-weight: bold; color: darkred;' if val == 'FLAG' else ''

        return df.style \
            .applymap(color_severity, subset=['severity']) \
            .applymap(highlight_status, subset=['status']) \
            .format({'amount': '${:,.2f}'})

    display_cols = ["timestamp", "txn_id", "user_id", "amount", "score", "severity", "status", "rules_triggered"]
    if "ai_explanation" in results_df.columns:
        display_cols.append("ai_explanation")

    st.header("Detailed Transaction Results")
    st.dataframe(style_fraud_df(results_df[display_cols]), use_container_width=True)

else:
    st.info("Please upload a CSV file to begin analysis.")
    st.markdown("""
        **Expected CSV columns:**  
        `txn_id`, `user_id`, `amount`, `currency`, `card_country`, `ip_country`,  
        `merchant_mcc`, `device_id`, `timestamp` (ISO 8601 format)
    """)
