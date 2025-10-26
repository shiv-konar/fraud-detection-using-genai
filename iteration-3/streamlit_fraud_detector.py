#!/usr/bin/env python3
# streamlit_fraud_detector.py
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
    r1 = rule_high_amount(t, config.get("high_amount_threshold", 2000.0))
    if r1: results.append(r1)

    r2 = rule_country_mismatch(t)
    if r2: results.append(r2)

    r3 = rule_merchant_blacklist(t, config.get("blacklisted_mcc", []))
    if r3: results.append(r3)

    recent = user_history.get(t.user_id, [])
    r4 = rule_velocity(t, recent, config.get("velocity_window_minutes", 5), config.get("velocity_max_allowed", 3))
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
