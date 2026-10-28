"""Post-hoc implementation/estimand audit; never modify original data or verdicts."""
import hashlib
import json
from pathlib import Path


def bound(pos: int, neg: int, n: int) -> dict:
    from scipy.stats import beta
    lp = 0. if pos == 0 else float(beta.ppf(.0125, pos, n-pos+1))
    up = 1. if pos == n else float(beta.ppf(.9875, pos+1, n-pos))
    lm = 0. if neg == 0 else float(beta.ppf(.0125, neg, n-neg+1))
    um = 1. if neg == n else float(beta.ppf(.9875, neg+1, n-neg))
    return {"N": n, "positive": pos, "negative": neg, "difference": (pos-neg)/n,
            "lower_975": lp-um, "upper_975": up-lm}


def main() -> None:
    import numpy as np
    from PIL import Image
    root = Path(__file__).parent
    folder = root / "v1"
    records = [json.loads((folder / "sources" / f"{i:04d}" / "completed.json").read_text())
               for i in range(1000)]
    rows = [json.loads((folder / "sources" / f"{i:04d}" / "rows.json").read_text())
            for i in range(1000)]
    flagged = [i for i, record in enumerate(records) if record["safety_replaced"]]
    original = json.loads((root / "final_native_audit" / "final_verdict.json").read_text())
    flags = original["dependency_flags"]
    flag_keys = {records[i]["prompt_key"] for i in flagged}
    assert len(flagged) == 44
    assert len(flag_keys) == 44
    assert len(flags) == 44*43//2
    assert all(f["payload"] == "v1" and f["type"] == "within_G_near"
               and f["a"] in flag_keys and f["b"] in flag_keys for f in flags)
    before_hashes, after_hashes = set(), set()
    replacement_matches = []
    asset = Image.open(root / "original_gpu" / "assets" / "rick.jpeg").convert("RGB")
    for j, i in enumerate(flagged):
        source = folder / "sources" / f"{i:04d}"
        with np.load(source / "native_boundaries.npz", allow_pickle=False) as data:
            before, after = data["before"], data["after"]
        before_hashes.add(hashlib.sha256(before.tobytes()).hexdigest())
        after_hashes.add(hashlib.sha256(after.tobytes()).hexdigest())
        normalized = np.asarray(asset.resize((before.shape[1], before.shape[0]))) / 255.
        candidates = [
            (255. * normalized.astype(dtype)).astype(np.uint8)
            for dtype in [np.float16, np.float32]
        ]
        replacement_matches.append(any(np.array_equal(before, candidate) for candidate in candidates))
        assert all(r["tau_accept"] == 0 and r["exact"] == 0 for r in rows[i])
        print(f"[SAFETY AUDIT] [{'#'*((j+1)*20//44):20s}] {j+1}/44 "
              "replacement boundary and zero-credit policy checked", flush=True)
    assert all(replacement_matches)
    summaries = []
    for condition in ["C0", "C444", "C420"]:
        cells = [next(r for r in rs if r["condition"] == condition) for rs in rows]
        selected = [cells[i] for i in flagged]
        clean = [cells[i] for i in range(1000) if i not in set(flagged)]
        summaries.append({
            "condition": condition, "N_attempts": 1000, "N_not_replaced": len(clean),
            "unconditional_tau": sum(r["tau_accept"] for r in cells)/1000,
            "decoder_only_tau": sum(r["decoder_tau_accept"] for r in cells)/1000,
            "conditional_nonreplaced_tau": sum(r["tau_accept"] for r in clean)/len(clean),
            "flagged_decoder_tau_count": sum(r["decoder_tau_accept"] for r in selected),
            "flagged_decoder_exact_count": sum(r["decoder_exact"] for r in selected),
            "flagged_K": sorted({r["K"] for r in selected}),
            "flagged_BER": selected[0]["BER"],
            "mean_BER_unconditional": sum(r["BER"] for r in cells)/1000,
            "mean_BER_nonreplaced": sum(r["BER"] for r in clean)/len(clean),
        })
    differences = [
        next(r["tau_accept"] for r in rs if r["condition"] == "C444")
        - next(r["tau_accept"] for r in rs if r["condition"] == "C420")
        for rs in rows
    ]
    assert all(differences[i] == 0 for i in flagged)
    clean = [d for i, d in enumerate(differences) if i not in set(flagged)]
    pos, neg = clean.count(1), clean.count(-1)
    result = {
        "status": "PURPOSE_AND_SAFETY_IMPACT_AUDIT_COMPLETE",
        "analysis_role": "Post-hoc diagnosis and sensitivity, not replacement of locked primary verdict",
        "safety_count": len(flagged), "safety_fraction": len(flagged)/1000,
        "distinct_flagged_prompt_groups": len(flag_keys),
        "all_flagged_boundaries_match_original_replacement_asset": all(replacement_matches),
        "flagged_unique_pre_watermark_arrays": len(before_hashes),
        "flagged_unique_post_watermark_arrays": len(after_hashes),
        "duplicate_pairs": len(flags), "all_pairs_explained_by_safety": True,
        "rates_and_BER": summaries,
        "primary_all_attempts": bound(pos, neg, 1000),
        "posthoc_nonreplaced_conditional": bound(pos, neg, len(clean)),
        "posthoc_adversarial_44_discordances": bound(pos, neg+len(flagged), 1000),
        "posthoc_unknown_prereplacement_difference_range": [
            (pos-neg-len(flagged))/1000, (pos-neg+len(flagged))/1000],
        "sensitivity_limit": "Bounds retain the original independent-prompt model; not a new proof of independence or an exact clustered interval.",
        "primary_verdict_preserved": original["verdicts"],
        "unobserved": "Pre-safety-check generated pixels were not saved. Distinct pre-safety images and correctness of safety classification cannot be audited from retained boundaries.",
    }
    out = root / "purpose_safety_impact_20261028"
    out.mkdir(exist_ok=False)
    (out / "safety_impact.json").write_text(json.dumps(result, indent=2))
    print("[AUDIT COMPLETE]", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
