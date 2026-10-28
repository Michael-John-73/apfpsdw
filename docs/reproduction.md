# Reproduction scope

## What we release

We release numerical native measurements and aggregate count-prediction
results alongside our authored component workflow and native scripts.
We keep original raw archives and execution fingerprints separate from
portable public exports.

We recorded Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1,
PyTorch 2.5.1+cu124, OpenCV 5.0.0, Pillow 12.3.0 and
invisible-watermark 0.2.0 in the prediction study. We recorded
PyWavelets runtime 1.8.0 with metadata 1.9.0. We do not assume that a newer
dependency installation reproduces the original environment byte-for-byte.

## Numerical verification

```bash
python tools/verify_release.py
python -m unittest discover -s tests -p "test_*.py" -v
```

We verify all public-file checksums and recompute native acceptance rates,
the six paired contrasts and their Clopper-Pearson difference bounds from
18,000 released rows. We retain the original audit hold independently of
whether numerical margins pass. This recount does not replay native images,
rerun generation, or establish prompt independence.

The original `revision_B/independent_recompute.py --run ARCHIVE` requires the
complete five-split raw archive, forecasts and fitted model records. We do
not include those image-derived feature records in this release. Aggregate
JSON results alone do not reproduce the original 409,504-row audit.

## New component experiments

We retain the original source layout so the staged launcher resolves its
modules and source fingerprint correctly:

```bash
python run_paper_b.py --help
python run_paper_b.py --config configs/component.template.json init
```

The template has a new study identifier, relative output paths and no source
images. Initialization creates a new run, not a reconstruction of our completed
experiment. Preparation requires the historical image/index layout expected
by `data.py`, fresh source paths and original-provider metadata. Calibration
and test evaluation require all preceding source-proof and forecast gates.
We preserve frozen groups, weights and decision criteria within each run.

## Native issuance

We used SDXL base 1.0 at revision
`462165984030d82259a11f4367a4eed129e94a7b`, and original SD v1-4 weights at
revision `f0bb45b49990512c454cf2c5670b0952ef2f9c71`, with CompVis code commit
`21f890f9da3cfbeaba8e2ac3c425ee9e998d5229`.

We used SDXL 1024-square, 30 steps, guidance 5.0, FP16 with VAE upcast;
original v1 used 512-square, PLMS50, guidance 7.5 and CUDA autocast.
We used seed 20261026 plus prompt index on an RTX 4090 (24 GB).
We retained the original v1 safety-check/replacement-before-watermark order.
The cohorts include all 44 safety replacements.

The `native/` scripts preserve original model-cache, manifest and frozen-run
requirements. Generation and final auditing accept external archive paths.
The safety diagnostic expects archived `v1/`, `final_native_audit/` and
`original_gpu/` directories beside the script.
We do not provide model weights, original CompVis code, prompt manifests,
safety assets or native-boundary arrays. They are research execution/audit
sources, not a self-contained one-command generation demo.

## Public-export provenance

We anonymized native source identifiers while preserving photograph pairing
and shared prompt indices across payloads. We retained every numerical
measurement, including safety flags and failed detections.
We excluded file/pixel hashes that identify unreleased image assets.

We revised explanatory output text in `operations.py` and made native
module imports repository-relative. We did not change numerical formulas
or decision rules. `source_provenance.json` retains original execution hashes;
the release manifest separately authenticates the exported bytes.
