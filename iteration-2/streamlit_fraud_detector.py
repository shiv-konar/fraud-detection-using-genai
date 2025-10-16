#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

@dataclass
class Transaction:
    txn_id: str
    user_id: str
    amount: float
    currency: str
    card_country: str
    ip_country: str
    merchant_mcc: int
    device_id: str
    timestamp: datetime

@dataclass
class RuleResult:
    name: str
    points: int
    details: str

def parse_timestamp(ts: str) -> datetime:
    # Expect ISO 8601 like "2025-10-06T14:35:00"
    return datetime.fromisoformat(ts)

def read_transactions(csv_path: str) -> List[Transaction]:
    txns: List[Transaction] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txns.append(Transaction(
                txn_id=row["txn_id"],
                user_id=row["user_id"],
                amount=float(row["amount"]),
                currency=row["currency"],
                card_country=row["card_country"],
                ip_country=row["ip_country"],
                merchant_mcc=int(row["merchant_mcc"]),
                device_id=row["device_id"],
                timestamp=parse_timestamp(row["timestamp"]),
            ))
    # Sort by timestamp to make velocity/device rules more sensible
    txns.sort(key=lambda t: t.timestamp)
    return txns

def rule_high_amount(t: Transaction, high_amount_threshold: float) -> RuleResult | None:
    if t.amount >= high_amount_threshold:
        points = 50 if t.amount >= high_amount_threshold * 2 else 30
        return RuleResult("HighAmount", points, f"Amount {t.amount:.2f} >= {high_amount_threshold}")
    return None

def rule_country_mismatch(t: Transaction) -> RuleResult | None:
    if t.card_country and t.ip_country and t.card_country != t.ip_country:
        return RuleResult("CountryMismatch", 40, f"Card country {t.card_country} != IP country {t.ip_country}")
    return None

def rule_merchant_blacklist(t: Transaction, blacklist: List[int]) -> RuleResult | None:
    if t.merchant_mcc in blacklist:
        return RuleResult("BlacklistedMCC", 25, f"MCC {t.merchant_mcc} is blacklisted")
    return None

def rule_velocity(t: Transaction, recent_txns_by_user: List[Transaction], window_minutes: int, max_allowed: int) -> RuleResult | None:
    window_start = t.timestamp - timedelta(minutes=window_minutes)
    count = sum(1 for rt in recent_txns_by_user if window_start <= rt.timestamp < t.timestamp)
    if count >= max_allowed:
        return RuleResult("HighVelocity", 30, f"{count} txns in last {window_minutes} minutes for user {t.user_id}")
    return None

def rule_night_time(t: Transaction) -> RuleResult | None:
    hour = t.timestamp.hour
    if 0 <= hour < 5:
        return RuleResult("NightTime", 10, f"Transaction at {hour:02d}:00")
    return None

def rule_new_device(t: Transaction, seen_devices: Dict[str, set]) -> RuleResult | None:
    devices = seen_devices.get(t.user_id, set())
    if devices and t.device_id not in devices:
        return RuleResult("NewDevice", 20, f"New device {t.device_id} for user {t.user_id}")
    return None

def score_transaction(
    t: Transaction,
    user_history: Dict[str, List[Transaction]],
    seen_devices: Dict[str, set],
    config: Dict
) -> Tuple[int, List[RuleResult]]:
    results: List[RuleResult] = []

    # Apply rules
    r1 = rule_high_amount(t, config["high_amount_threshold"])
    if r1: results.append(r1)

    r2 = rule_country_mismatch(t)
    if r2: results.append(r2)

    r3 = rule_merchant_blacklist(t, config["blacklisted_mcc"])
    if r3: results.append(r3)

    recent = user_history.get(t.user_id, [])
    r4 = rule_velocity(t, recent, config["velocity_window_minutes"], config["velocity_max_allowed"])
    if r4: results.append(r4)

    r5 = rule_night_time(t)
    if r5: results.append(r5)

    r6 = rule_new_device(t, seen_devices)
    if r6: results.append(r6)

    total_points = sum(r.points for r in results)
    return total_points, results

def severity_label(score: int) -> str:
    if score >= 80: return "HIGH"
    if score >= 50: return "MEDIUM"
    if score >= 20: return "LOW"
    return "INFO"


def run(csv_path: str, flag_threshold: int, config: Dict) -> List[Dict]:
    # Streamlit will use io.StringIO instead of a physical path,
    # so we need to handle that flexibility in read_transactions (see notes below)
    # For now, if you are running from an actual path, this works:

    # You might need to adjust read_transactions to accept a file-like object
    # for the Streamlit version, but for a simple path run, this is fine.

    txns = read_transactions(csv_path)

    user_history: Dict[str, List[Transaction]] = {}
    seen_devices: Dict[str, set] = {}

    # List to hold the results for the UI
    results = []

    for t in txns:
        score, rule_results = score_transaction(t, user_history, seen_devices, config)
        sev = severity_label(score)
        flagged = "FLAG" if score >= flag_threshold else "OK"

        # Prepare a dictionary for this transaction's result
        result_entry = {
            "txn_id": t.txn_id,
            "user_id": t.user_id,
            "amount": t.amount,
            "currency": t.currency,
            "card_country": t.card_country,
            "ip_country": t.ip_country,
            "timestamp": t.timestamp.isoformat(),
            "score": score,
            "severity": sev,
            "status": flagged,
            "rules_triggered": [
                {"name": rr.name, "points": rr.points, "details": rr.details}
                for rr in rule_results
            ]
        }
        results.append(result_entry)

        # Update history and seen devices after scoring
        user_history.setdefault(t.user_id, []).append(t)
        seen_devices.setdefault(t.user_id, set()).add(t.device_id)

    return results