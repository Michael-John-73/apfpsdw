# Native evidence and chronology

We link the manuscript's native contrast and acceptance tables (Tables 4 and 5)
to the released numerical records. We distinguish archived execution evidence
from external preregistration and from a complete replay of image generation.

## Table-to-record mapping

| Manuscript evidence | Released record | Verification |
|---|---|---|
| Table 4: two S-F and four S-T contrasts | [summary contrasts](../data/native/summary.json), [paired measurements](../data/native/measurements.jsonl) | We pair `group` identifiers within payload and contrast, recount positive/negative discordances and recompute the CP bounds |
| Table 5: exact/tolerant rates in 18 cells | [summary rates](../data/native/summary.json), [measurements](../data/native/measurements.jsonl) | We recount 1,000 observations per payload/role/channel cell from bit-error counts |
| 44 v1 safety replacements | `safety_replaced` and `group` in the measurements; [post-hoc analysis](../data/native/safety.json) | We check all 132 channel rows have K=68 and reject; we retain all attempts and the original S-T hold |

From the repository root, we run:

```bash
python tools/verify_release.py
python -m unittest discover -s tests -p "test_*.py" -v
```

These commands recount exported measurements, not source images. The original
asset-match diagnosis and image-boundary replay require the separate archive.
We do not infer prompt independence from a successful numerical recount.

## Recorded chronology

| Stage | Recorded anchor | What we can establish from the record | Exposure or limitation |
|---|---|---|---|
| Initial preparation | `experiment_start_status.json`, embedded UTC 2026-10-26 12:30:33.316104 | S-F/S-T margins 0.05/0.10, alpha 0.025 each, channel contrast and IUT rule are recorded | N=1,000 remains a candidate; this is not the final sample lock |
| RF/R0 measurement | Run label `rf_experiment_20261026T130533Z`; `prelabel_lock.json` and `rf_results.json` | The lock records N=1,000, source/script hashes, channels, margins and inference rules; the result's recorded lock hash matches the archived lock bytes | The run label is not an independently certified timestamp; the lock has no embedded creation time |
| Earlier CPU issuance | Retired cohorts, before the GPU-only restart | We exclude 108 XL and 237 v1 CPU outputs from final native estimates | The shared prompt list was already partially exposed; restarting does not undo exposure |
| GPU-only native issuance | Archive label `gpu_only_20261028`; payload-specific `prelabel_lock.json` | Both locks record N=1,000, the same prompt-manifest hash, model revision, seeds, generation settings and prior CPU exposure | RF measurements preceded the native configuration lock; the lock files do not establish an independently certified creation time |
| Runtime and final audit | Payload-specific `runtime_lock.json`; `final_native_audit/final_verdict.json` | Runtime records identify XL pipeline/watermarker hashes and original v1 code/checkpoint; the final audit retains both generation-lock contents | Safety-checker weight provenance is less complete; batch/source execution updates remain distinct from scientific settings |

We verified that the RF result references the archived lock hash and that
the final native audit embeds the payload-specific generation-lock contents.
The archived records support an explicit account of settings and exposure,
not proof that every item was irreversibly fixed before every observation.
We do not present this extension as externally preregistered or wholly
unexposed. The prior exposure and fixed RF-before-native order remain
limitations rather than reasons to replace the original verdict.

### Archived record SHA-256

These hashes identify the archived records, not modified public exports.
We omit prompts, image identities, connection details and original images.

| Archived record | SHA-256 |
|---|---|
| Initial preparation status | `1cea77b3ad862eea52dec02bcf186d8bf52b9fcd9c454f42724831432c429dfc` |
| RF prelabel lock | `cd86c9691cec12c48d846c9f47a3551049aff87573eb66f2579e2d308011d0c2` |
| RF result | `55e375ec98b1f9e822c897e0f69d29ccbd1ae069ca88fcee9169f2b73a8af61d` |
| GPU-only XL prelabel lock | `35a09fcd0f332618882edfa770d2722dbef17f0d1be4559909f6cefbfb1977f0` |
| GPU-only v1 prelabel lock | `b3b081ef688b8e3c9d429a97004df110f080c1139b0a78210c69cc3916b4ff20` |
| XL runtime lock | `07dd181debe3e32aa3b3e2d8683cc2eaf66e456c532306c3f7473ad4ae8f2a55` |
| v1 runtime lock | `3fbc174299cb79448d3128d22df9317c18fef3e303297c3c14de3ed9a7833dc9` |

## Model, code and environment identification

We list generator revisions and reproduction prerequisites in
[reproduction scope](reproduction.md). The XL runtime lock records Diffusers
0.40.0, pipeline source SHA-256
`9a5a1d2dd98f38b0171a1bffd39d80e93bc1a8bdbeea5f36444f383c067d286b`
and watermarker source SHA-256
`2c322dbd19d92a890851c84fd2822b3b6130f400118403f830cad063c99a43a4`.
The original v1 runtime lock records CompVis commit
`21f890f9da3cfbeaba8e2ac3c425ee9e998d5229` and checkpoint SHA-256
`fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556`.
We do not claim equally complete safety-checker weight identification.

For the separate prediction study, we collected PyWavelets runtime 1.8.0
from `pywt.__version__`, whereas the environment report collected distribution
metadata 1.9.0 using `importlib.metadata.version`. The archived fingerprint
does not identify the imported PyWavelets path or compiled-module hash.
We retain the discrepancy as unresolved; neither version string alone
establishes the exact dependency bytes. A new local installation cannot
retroactively supply that missing execution evidence.
