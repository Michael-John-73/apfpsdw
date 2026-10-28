"""Independent full-cohort audit; refuse a final verdict while any G source is missing."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def read(path: Path) -> object:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def bounds(positive: int, negative: int, n: int) -> tuple[float, float]:
    from scipy.stats import beta
    lp = 0. if positive == 0 else float(beta.ppf(.0125, positive, n-positive+1))
    up = 1. if positive == n else float(beta.ppf(.9875, positive+1, n-positive))
    lm = 0. if negative == 0 else float(beta.ppf(.0125, negative, n-negative+1))
    um = 1. if negative == n else float(beta.ppf(.9875, negative+1, n-negative))
    return lp-um, up-lm


def audit(args: argparse.Namespace) -> None:
    import numpy as np
    from PIL import Image
    from imwatermark import WatermarkDecoder
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "revision_B" / "src"))
    from paper_b.codec import channel, PAYLOADS, encode

    rows = []
    group_hashes = {}
    phashes = {}
    source_receipts = {}
    native_locks = {}
    execution_updates = {}
    for payload, folder in [("xl", args.xl), ("v1", args.v1)]:
        lock = read(folder / "prelabel_lock.json")
        if not isinstance(lock, dict) or lock["N"] != 1000:
            raise RuntimeError("Invalid native cohort lock")
        native_locks[payload] = lock
        authorization_path = folder / "resume_authorization.json"
        if lock.get("phase") == "MAIN_NATIVE_GPU_ONLY" and authorization_path.exists():
            authorization = read(authorization_path)
            execution = read(folder / authorization["execution_lock"])
            if (authorization["old_source_sha256"] != lock["source_sha256"]
                    or authorization["new_source_sha256"] != execution["source_sha256"]
                    or execution["generation"] != lock["generation"]
                    or execution["prompt_manifest_sha256"] != lock["prompt_manifest_sha256"]):
                raise RuntimeError("GPU execution update does not preserve scientific protocol")
            execution_updates[payload] = {"authorization": authorization, "execution_lock": execution}
        sources = folder / "sources"
        records = []
        if len(list(sources.glob("*/completed.json"))) != 1000:
            raise RuntimeError(f"FINAL VERDICT BLOCKED: {payload} does not have 1000 committed native sources")
        marker = StableDiffusionXLWatermarker() if payload == "xl" else None
        decoder = WatermarkDecoder("bits", len(PAYLOADS[payload]))
        for index in range(1000):
            source = sources / f"{index:04d}"
            record = read(source / "completed.json")
            if not isinstance(record, dict) or record["index"] != index or not record["native_replay_equal"]:
                raise RuntimeError("Invalid native source receipt")
            for name, expected in record["files"].items():
                if sha(source / name) != expected:
                    raise RuntimeError("Native source evidence hash differs")
            issued = np.asarray(Image.open(source / "issued.png").convert("RGB"))
            with np.load(source / "native_boundaries.npz", allow_pickle=False) as arrays:
                before, after = arrays["before"], arrays["after"]
            if payload == "xl":
                if marker is None:
                    raise RuntimeError("Native marker not initialized")
                replay = marker.apply_watermark(torch.from_numpy(before.copy()))
                if not np.array_equal(replay.numpy(), after):
                    raise RuntimeError("Independent full SDXL boundary replay differs")
                rgb = np.clip(np.rint((after[0].transpose(1, 2, 0) / 2 + .5) * 255), 0, 255).astype(np.uint8)
            else:
                # Reused codec was checked against deployed encoder during RF study;
                # this independent engine also checks native byte-array insertion.
                bgr, _ = encode(before[:, :, ::-1].copy(), PAYLOADS["v1"], 36., "uint8", True)
                rgb = bgr[:, :, ::-1]
                if not np.array_equal(rgb, after):
                    raise RuntimeError("Independent V1 native insertion replay differs")
            if not np.array_equal(rgb, issued):
                raise RuntimeError("Independent final native image replay differs")
            source_rows = read(source / "rows.json")
            if not isinstance(source_rows, list) or len(source_rows) != 3:
                raise RuntimeError("Missing native channel rows")
            flags = bool(record.get("safety_replaced", False))
            for row in source_rows:
                if row["payload"] != payload or row["role"] != "G" or row["prompt_key"] != record["prompt_key"]:
                    raise RuntimeError("Native row provenance mismatch")
                stages = [] if row["condition"] == "C0" else [
                    {"op": "jpeg", "quality": 95, "subsampling": 0 if row["condition"] == "C444" else 2}]
                image = channel(issued[:, :, ::-1].copy(), stages, str(index))
                bits = decoder.decode(image, "dwtDct").astype(np.uint8)
                k = int((bits != PAYLOADS[payload]).sum())
                if row["K"] != k or row["tau_accept"] != int(k <= len(bits)//8 and not flags):
                    raise RuntimeError("Independent G decoding mismatch")
                rows.append(dict(row, source_group=row["prompt_key"]))
            records.append(record)
            if (index+1) % 50 == 0:
                print(f"[FINAL AUDIT] {payload}: {index+1}/1000 native sources independently replayed/decoded", flush=True)
        group_hashes[payload] = {r["prompt_key"]: r["pixel_sha256"] for r in records}
        phashes[payload] = [(r["prompt_key"], int(r["phash"])) for r in records]
        source_receipts[payload] = [sha(sources / f"{i:04d}" / "completed.json") for i in range(1000)]
    if set(group_hashes["xl"]) != set(group_hashes["v1"]):
        raise RuntimeError("Two generators do not share exactly the same 1000 prompt groups")
    gpu_only = [lock.get("phase") == "MAIN_NATIVE_GPU_ONLY" for lock in native_locks.values()]
    if any(gpu_only) and not all(gpu_only):
        raise RuntimeError("Mixing retired CPU and GPU-only native cohorts is forbidden")
    native_scope = (
        "GPU-only restart: XL fp16 with official VAE upcast; original V1 CUDA autocast. "
        "All CPU-native rows excluded. Batch sizes chosen only by VRAM telemetry; "
        "same prompts previously partly exposed on CPU. No universal SD release claim."
        if all(gpu_only) else "CPU float32 declarations only; no universal SD release claim."
    )
    rf_rows = [json.loads(line) for line in (args.rf / "rows.jsonl").read_text().splitlines()]
    rf_result = read(args.rf / "rf_results.json")
    if not isinstance(rf_result, dict) or sha(args.rf / "rows.jsonl") != rf_result["rows_sha256"] or len(rf_rows) != 12000:
        raise RuntimeError("RF evidence mismatch")
    rows.extend(rf_rows)
    keys = {(r["source_group"], r["payload"], r["role"], r["condition"]) for r in rows}
    if len(rows) != 18000 or len(keys) != 18000:
        raise RuntimeError("Expected exactly 18000 unique measurement cells")
    contrasts = []
    for payload in ["xl", "v1"]:
        for name, role, left, right, margin in [
            ("S-F", "RF_vs_R0", ("RF", "C0"), ("R0", "C0"), .05),
            ("S-T", "RF", ("RF", "C444"), ("RF", "C420"), .10),
            ("S-T", "G", ("G", "C444"), ("G", "C420"), .10),
        ]:
            a = {r["source_group"]: r["tau_accept"] for r in rows if r["payload"] == payload and (r["role"], r["condition"]) == left}
            b = {r["source_group"]: r["tau_accept"] for r in rows if r["payload"] == payload and (r["role"], r["condition"]) == right}
            if set(a) != set(b) or len(a) != 1000:
                raise RuntimeError("Invalid full-cohort pairing")
            d = [a[k]-b[k] for k in a]
            pos, neg = d.count(1), d.count(-1)
            lo, hi = bounds(pos, neg, 1000)
            contrasts.append({"hypothesis": name, "population": role, "payload": payload, "n": 1000,
                              "difference": sum(d)/1000, "lower_975": lo, "upper_975": hi, "margin": margin,
                              "positive_discordant": pos, "negative_discordant": neg,
                              "verdict": "SUPPORTED" if lo > margin else ("BELOW_MARGIN" if hi < margin else "INCONCLUSIVE")})
    # Preserve all generated outputs; no favourable dropping/replacement of duplicates.
    duplicates = []
    for payload, pairs in phashes.items():
        for i, (key, value) in enumerate(pairs):
            for other, old in pairs[:i]:
                if (value ^ old).bit_count() <= 4:
                    duplicates.append({"payload": payload, "a": key, "b": other, "type": "within_G_near"})
    past = read(args.historical)
    if not isinstance(past, list):
        raise RuntimeError("Invalid historical source manifest")
    prior_phashes = [int(r["phash"]) for r in past if r["split"] not in ["reserve", "reserved_fresh"]]
    for payload, pairs in phashes.items():
        for key, value in pairs:
            if any((value ^ old).bit_count() <= 4 for old in prior_phashes):
                duplicates.append({"payload": payload, "a": key, "type": "near_past_exposure"})
    # Unresolved dependence cannot be erased by reporting 18000 rows as N.
    verdicts = {}
    for name in ["S-F", "S-T"]:
        cells = [r for r in contrasts if r["hypothesis"] == name]
        verdicts[name] = ("INCONCLUSIVE_DEPENDENCE_AUDIT" if duplicates and name == "S-T" else
                         ("SUPPORTED" if all(r["verdict"] == "SUPPORTED" for r in cells)
                          else "BELOW_MARGIN_IN_AT_LEAST_ONE_CONTRAST" if any(r["verdict"] == "BELOW_MARGIN" for r in cells)
                          else "INCONCLUSIVE"))
    rates = []
    for payload in ["xl", "v1"]:
        for role in ["G", "R0", "RF"]:
            for condition in ["C0", "C444", "C420"]:
                selected = [r for r in rows if (r["payload"], r["role"], r["condition"]) == (payload, role, condition)]
                if len(selected) != 1000:
                    raise RuntimeError("Missing joint operating-rate cell")
                rates.append({"payload": payload, "role": role, "condition": condition, "n": 1000,
                              "exact": sum(r["exact"] for r in selected)/1000,
                              "tau_accept": sum(r["tau_accept"] for r in selected)/1000,
                              "BER": sum(r["BER"] for r in selected)/1000,
                              "safety_replaced": sum(bool(r.get("safety_replaced", False)) for r in selected)})
    result = {"status": "FULL_COHORT_INDEPENDENT_AUDIT_COMPLETE", "rows": 18000,
              "independent_units": {"natural_photo_groups": 1000, "shared_prompt_groups": 1000,
                                    "not_4000_independent_groups": True},
              "verdicts": verdicts, "contrasts": contrasts, "rates": rates, "dependency_flags": duplicates,
              "insertion_quality": rf_result["quality"],
              "native_source_receipt_hashes": source_receipts,
              "native_generation_locks": native_locks,
              "native_execution_updates": execution_updates,
              "original_H1_H2_H3_HBP_HBT": "UNCHANGED",
              "limitations": [native_scope,
                             "N fixed to report target without native-G power guarantee.",
                             "One-sided per-contrast bounds are not simultaneous intervals.",
                             "RF results were already exposed before G configuration lock; chronology is retained."]}
    write_json = args.output / "final_verdict.json"
    args.output.mkdir(exist_ok=False, parents=True)
    write_json.write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ["# Native-linked supplemental experiments: final audit", "",
             "Completed 1000 natural-photo groups and 1000 shared prompt groups; 18000 repeated measurement rows.",
             "Both actual native issuance paths were independently replayed and decoded.", "",
             "## Final verdict", "", "| Hypothesis | Verdict |", "|---|---|"]
    lines += [f"| {name} | {verdict} |" for name, verdict in verdicts.items()]
    lines += ["", "## Contrasts", "", "| Hypothesis | Population | Payload | Difference | One-sided 97.5% lower | Margin | Verdict |",
              "|---|---|---|---:|---:|---:|---|"]
    lines += [f"| {r['hypothesis']} | {r['population']} | {r['payload']} | {r['difference']:.6f} | {r['lower_975']:.6f} | {r['margin']:.2f} | {r['verdict']} |" for r in contrasts]
    lines += ["", "## Native detection, natural false positives and public-insertion acceptance", "",
              "| Payload | Population | Channel | N | Exact | Tau | BER | Safety replacement count |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['payload']} | {r['role']} | {r['condition']} | {r['n']} | {r['exact']:.6f} | {r['tau_accept']:.6f} | {r['BER']:.6f} | {r['safety_replaced']} |" for r in rates]
    lines += ["", "## Insertion-only quality", "", "| Payload | Mean PSNR (dB) | Mean RGB SSIM |", "|---|---:|---:|"]
    lines += [f"| {payload} | {q['mean_finite_psnr']:.6f} | {q['mean_ssim']:.6f} |" for payload, q in rf_result["quality"].items()]
    lines += ["", "## Scope and limits", "",
              "Original H1/H2/H3 and HBP/HBT decisions are unchanged. No new training or human perception study.",
              "Source dependence flags block independent-group confirmatory claims when present.",
              native_scope,
              "N was the fixed report target, not guaranteed native-G power.",
              "RF measurements preceded the native-G configuration lock; this chronology is not concealed.",
              "", "[Machine-readable audit and verdict](final_verdict.json)"]
    (args.output / "supplemental_native_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FINAL VERDICT] {verdicts}; persisted at {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for field in ["xl", "v1", "rf", "historical", "output"]:
        parser.add_argument("--" + field, type=Path, required=True)
    audit(parser.parse_args())
