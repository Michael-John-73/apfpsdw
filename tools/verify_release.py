"""Verify exported checksums and independently recount native measurements."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_bounds(positive: int, negative: int, n: int) -> tuple[float, float]:
    if n <= 0 or min(positive, negative) < 0 or positive + negative > n:
        raise ValueError("Invalid discordant-pair counts")
    lp = 0.0 if positive == 0 else float(beta.ppf(.0125, positive, n-positive+1))
    up = 1.0 if positive == n else float(beta.ppf(.9875, positive+1, n-positive))
    lm = 0.0 if negative == 0 else float(beta.ppf(.0125, negative, n-negative+1))
    um = 1.0 if negative == n else float(beta.ppf(.9875, negative+1, n-negative))
    return lp-um, up-lm


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, name: str) -> None:
    require(abs(actual-expected) < 1e-12, f"{name}: {actual} != {expected}")


def verify_checksums() -> int:
    manifest = read(ROOT / "data/release_manifest.json")
    for relative, expected in manifest["sha256"].items():
        path = (ROOT / relative).resolve()
        require(ROOT in path.parents, f"Manifest path outside release: {relative}")
        require(path.is_file(), f"Missing released file: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"Checksum mismatch: {relative}")
    return len(manifest["sha256"])


def verify_native() -> dict:
    rows = [
        json.loads(line)
        for line in (ROOT / "data/native/measurements.jsonl").read_text().splitlines()
    ]
    require(len(rows) == 18000, "Expected 18,000 native measurement cells")
    summary = read(ROOT / "data/native/summary.json")
    cells = defaultdict(dict)
    for row in rows:
        nbits = 48 if row["payload"] == "xl" else 136
        require(row["L"] == nbits, "Payload length mismatch")
        require(type(row["K"]) is int and 0 <= row["K"] <= nbits, "Invalid bit errors")
        require(row["exact"] == int(row["K"] == 0), "Exact decision mismatch")
        require(row["tau_accept"] == int(row["K"] <= nbits//8), "Tolerant decision mismatch")
        close(row["BER"], row["K"]/nbits, "BER")
        key = (row["payload"], row["role"], row["condition"])
        require(row["group"] not in cells[key], f"Duplicate measurement: {key}")
        cells[key][row["group"]] = row
    require(len(cells) == 18, "Expected 18 payload/role/channel cells")
    require(len(summary["rates"]) == 18, "Expected 18 summary rate cells")
    for rate in summary["rates"]:
        cell = cells[(rate["payload"], rate["role"], rate["condition"])]
        require(len(cell) == rate["n"] == 1000, "Incorrect cell denominator")
        for metric in ("exact", "tau_accept", "BER"):
            close(sum(row[metric] for row in cell.values())/1000, rate[metric], metric)
    for contrast in summary["contrasts"]:
        payload = contrast["payload"]
        if contrast["hypothesis"] == "S-F":
            left = cells[(payload, "RF", "C0")]
            right = cells[(payload, "R0", "C0")]
        else:
            left = cells[(payload, contrast["population"], "C444")]
            right = cells[(payload, contrast["population"], "C420")]
        require(left.keys() == right.keys(), "Paired group identifiers differ")
        differences = Counter(
            left[group]["tau_accept"]-right[group]["tau_accept"] for group in left
        )
        positive, negative = differences[1], differences[-1]
        require(positive == contrast["positive_discordant"], "Positive discordance mismatch")
        require(negative == contrast["negative_discordant"], "Negative discordance mismatch")
        close((positive-negative)/1000, contrast["difference"], "Paired difference")
        low, high = paired_bounds(positive, negative, 1000)
        close(low, contrast["lower_975"], "Paired lower bound")
        close(high, contrast["upper_975"], "Paired upper bound")
    require(len(summary["contrasts"]) == 6, "Expected six paired contrasts")
    flagged = [row for row in rows if row["safety_replaced"]]
    require(len(flagged) == 44*3, "Expected 44 safety replacements in three channels")
    require(all(row["payload"] == "v1" and row["role"] == "G"
                and row["K"] == 68 and row["tau_accept"] == 0 for row in flagged),
            "Safety-replacement decisions differ")
    safety = read(ROOT / "data/native/safety.json")
    require(safety["duplicate_pairs"] == 44*43//2, "Safety duplicate-pair count differs")
    require(summary["verdicts"]["S-F"] == "SUPPORTED", "S-F status differs")
    require(summary["verdicts"]["S-T"] == "INCONCLUSIVE_DEPENDENCE_AUDIT",
            "Original S-T audit hold was not retained")
    return {"rows": len(rows), "cells": len(cells), "contrasts": 6}


def verify_prediction_summary() -> None:
    evaluation = read(ROOT / "data/prediction/evaluation.json")
    primary = evaluation["panels"]["primary"]
    require(primary["verdict"] == "INCONCLUSIVE", "Primary status differs")
    gain = primary["paired_nll_gain"]
    require(gain["low"] < primary["delta_nll"] < gain["estimate"],
            "Primary gain/margin relationship differs")
    cost = read(ROOT / "data/prediction/cost.json")["summary"]
    require(cost["predict"]["sum_s"] > cost["replay"]["sum_s"],
            "Prediction/replay timing relationship differs")


if __name__ == "__main__":
    count = verify_checksums()
    native = verify_native()
    verify_prediction_summary()
    print(f"PASS: {count} file checksums; {native}; original verdicts retained.")
    print("We recount released numerical records, not image generation or the full prediction archive.")
