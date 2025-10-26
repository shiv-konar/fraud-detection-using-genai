# llm_fraud_assistant.py
from typing import Dict, List, Optional
import os
import json
from openai import OpenAI
from dataclasses import asdict

# Create the client. The OpenAI Python SDK reads from env by default, but
# Streamlit or other hosts can inject into env before run.
client = OpenAI()

def _format_txn_for_prompt(txn: Dict) -> str:
    # txn is a small dict with the most useful fields
    return json.dumps(txn, indent=2, default=str)

def explain_fraud_decision(
    txn: Dict,
    rules_triggered: List[Dict],
    severity: str,
    score: int,
    max_tokens: int = 180,
    temperature: float = 0.4
) -> str:
    """
    Ask the LLM for a concise explanation suitable for UI display.
    txn: small dict (txn_id, user_id, amount, currency, card_country, ip_country, timestamp)
    rules_triggered: list of dicts with 'name', 'points', 'details'
    """
    txn_str = _format_txn_for_prompt(txn)
    rules_str = "\n".join([f"- {r['name']} (+{r['points']}): {r['details']}" for r in rules_triggered]) if rules_triggered else "None"

    prompt = f"""
You are a fraud detection analyst assistant. Produce a concise (1-3 sentence) explanation suitable for showing in a web UI, explaining why this transaction scored {score} ({severity}).

Transaction:
{txn_str}

Rules triggered:
{rules_str}

Do not invent facts. If the evidence is inconclusive, say the transaction 'looks suspicious' and recommend further steps (e.g., "verify user identity" or "request additional authentication").
"""

    # Use Chat Completions via client.chat.completions.create per the OpenAI Python SDK.
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # change to available model if necessary
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # The SDK returns choices with message content
    try:
        content = resp.choices[0].message.content.strip()
        return content
    except Exception:
        # Fallback: return a very short generated message or empty
        return "Explanation unavailable (LLM error)."

def summarize_flagged_transactions(flat_results: List[Dict], top_n: int = 5) -> Optional[str]:
    """
    Summarize patterns across flagged transactions.
    `flat_results` is a list of result dicts (same shape as shown in the UI).
    """
    flagged = [r for r in flat_results if r.get("status") == "FLAG"]
    if not flagged:
        return "No flagged transactions to summarize."

    # Prepare a short CSV-like snippet for context (truncate for token limits)
    # We'll include Up to top_n rows
    snippet_lines = []
    for r in flagged[:top_n]:
        snippet_lines.append(f"{r.get('txn_id')} | {r.get('user_id')} | {r.get('score')} | {r.get('timestamp')}")

    prompt = f"""
You are an analyst. Based on these flagged transactions, give a short summary (2-4 sentences) of possible patterns, e.g., "cross-border purchases", "high amounts", "device churn", "velocity spikes", and suggest up to 3 next steps for investigation.

Data sample:
{chr(10).join(snippet_lines)}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25,
        max_tokens=220,
    )
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return None
