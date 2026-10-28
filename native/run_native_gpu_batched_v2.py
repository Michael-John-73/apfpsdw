"""GPU-only native issuance with outcome-blind, memory-bounded production batches."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "revision_B" / "src"))
os.environ["HF_HUB_DISABLE_XET"] = "1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def locked_record(path: Path, value: object) -> None:
    def normalized(item):
        if isinstance(item, dict):
            return {k: sorted(v) if k == "_use_default_values" else normalized(v) for k, v in item.items()}
        if isinstance(item, list):
            return [normalized(v) for v in item]
        return item

    if path.exists():
        if normalized(json.loads(path.read_text())) != normalized(value):
            raise RuntimeError(f"Frozen GPU runtime record differs: {path.name}")
    else:
        save(path, value)


def next_batch(current: int, peak: int, baseline: int, budget: int, ceiling: int) -> int:
    if current < 1 or peak < baseline or budget <= 0:
        raise ValueError("Invalid memory telemetry")
    variable = peak - baseline
    capacity = ceiling if variable == 0 else int((budget - baseline) * current / variable)
    return max(1, min(current * 2, ceiling, capacity))


def load_xl(base: Path, output: Path):
    import numpy as np
    import torch
    import diffusers
    from diffusers import StableDiffusionXLPipeline

    snapshot = base / "cpu_generation/model_cache/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
    checkpoint = json.loads((base / "g_main_20261026T135202Z/checkpoint_manifest.json").read_text())
    for entry in checkpoint["files"]:
        if sha(snapshot / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Pinned XL model file differs: {entry['path']}")
    locked_record(output / "checkpoint_manifest.json", checkpoint)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(snapshot), variant="fp16", torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True, add_watermarker=True,
    ).to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    from paper_b.codec import PAYLOADS
    if pipe.watermark is None or list(map(int, pipe.watermark.watermark)) != PAYLOADS["xl"].tolist():
        raise RuntimeError("Official native XL watermark missing")
    original = pipe.watermark.apply_watermark
    captured = {}

    def capture(images):
        before = images.detach().clone()
        after = original(images)
        replay = original(before.clone())
        if not torch.equal(after, replay):
            raise RuntimeError("GPU XL native boundary replay differs")
        captured["before"] = before.cpu().numpy()
        captured["after"] = after.detach().cpu().numpy()
        return after

    pipe.watermark.apply_watermark = capture
    locked_record(output / "runtime_lock.json", {
        "scheduler_class": type(pipe.scheduler).__name__, "scheduler": dict(pipe.scheduler.config),
        "pipeline_source": sha(Path(inspect.getfile(type(pipe)))),
        "watermarker_source": sha(Path(inspect.getfile(type(pipe.watermark)))),
        "diffusers": diffusers.__version__, "attention": "default SDPA",
        "model_dtype": "float16", "vae": "official force-upcast, slicing and tiling",
    })

    def generate(prompts, indices):
        captured.clear()
        generators = [torch.Generator("cuda").manual_seed(20261026 + i) for i in indices]

        def progress(pipeline, step, timestep, values):
            print(f"[GPU XL] indices={indices[0]}..{indices[-1]} step={step+1}/30", flush=True)
            return values

        with torch.inference_mode():
            result = pipe(
                prompt=prompts, width=1024, height=1024, num_inference_steps=30,
                guidance_scale=5., num_images_per_prompt=1, generator=generators,
                callback_on_step_end=progress,
            )
        if set(captured) != {"before", "after"} or len(result.images) != len(indices):
            raise RuntimeError("Incomplete native GPU batch")
        before, after = captured.pop("before"), captured.pop("after")
        replay_pil = pipe.image_processor.postprocess(torch.from_numpy(after.copy()), output_type="pil")
        if any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(result.images, replay_pil)):
            raise RuntimeError("XL PIL boundary replay differs")
        return [(before[j:j+1], after[j:j+1], image, False) for j, image in enumerate(result.images)]

    return generate


def load_v1(base: Path, repo: Path, output: Path):
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from imwatermark import WatermarkEncoder

    checkpoint = base / "v1_model_cache/models--CompVis--stable-diffusion-v-1-4-original/snapshots/f0bb45b49990512c454cf2c5670b0952ef2f9c71/sd-v1-4.ckpt"
    if sha(checkpoint) != "fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556":
        raise RuntimeError("Pinned original V1 checkpoint differs")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    module = runpy.run_path(str(repo / "scripts/txt2img.py"), run_name="native_gpu_module")
    config = OmegaConf.load(repo / "configs/stable-diffusion/v1-inference.yaml")
    config.model.params.cond_stage_config.params = {"device": "cuda"}
    model = module["load_model_from_config"](config, str(checkpoint))
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Original model did not load onto CUDA")
    from ldm.models.diffusion.plms import PLMSSampler
    sampler = PLMSSampler(model)
    encoder = WatermarkEncoder()
    encoder.set_watermark("bytes", b"StableDiffusionV1")
    locked_record(output / "runtime_lock.json", {
        "checkpoint_sha256": sha(checkpoint), "repo_commit": "21f890f9da3cfbeaba8e2ac3c425ee9e998d5229",
        "original_sources": {name: sha(repo / name) for name in [
            "scripts/txt2img.py", "ldm/models/diffusion/plms.py", "ldm/models/diffusion/ddim.py"]},
        "weights": "float32", "compute": "official CUDA autocast float16; original PLMS50",
        "initial_noise": "one CUDA float32 generator per fixed per-prompt seed; supplied as x_T",
    })

    def generate(prompts, indices):
        noise = torch.cat([
            torch.randn((1, 4, 64, 64), device="cuda",
                        generator=torch.Generator("cuda").manual_seed(20261026 + i))
            for i in indices
        ])
        with torch.no_grad(), torch.autocast("cuda"), model.ema_scope():
            unconditional = model.get_learned_conditioning([""] * len(indices))
            conditional = model.get_learned_conditioning(prompts)

            def progress(step):
                print(f"[GPU V1] indices={indices[0]}..{indices[-1]} step={step+1}/50", flush=True)

            samples, _ = sampler.sample(
                S=50, conditioning=conditional, batch_size=len(indices), shape=[4, 64, 64],
                verbose=False, unconditional_guidance_scale=7.5,
                unconditional_conditioning=unconditional, eta=0., x_T=noise, callback=progress,
            )
            # Serial VAE decode prevents its activations from limiting denoising batches.
            images = [torch.clamp((model.decode_first_stage(sample[None]) + 1.) / 2., 0., 1.)
                      .cpu().permute(0, 2, 3, 1).numpy() for sample in samples]
        checked, flags = module["check_safety"](np.concatenate(images))
        outputs = []
        for image, flagged in zip(checked, flags):
            before = (255. * image).astype(np.uint8)
            issued = module["put_watermark"](Image.fromarray(before), encoder)
            replay = module["put_watermark"](Image.fromarray(before.copy()), encoder)
            if not np.array_equal(np.asarray(issued), np.asarray(replay)):
                raise RuntimeError("Original V1 native replay differs")
            outputs.append((before, np.asarray(issued), issued, bool(flagged)))
        return outputs

    return generate


def run(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    import torch
    from imwatermark import WatermarkDecoder
    from paper_b.codec import PAYLOADS, channel, decode
    from paper_b.data import perceptual_hash

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; CPU fallback forbidden")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prompts = json.loads(args.manifest.read_text())
    if len(prompts) != 1000 or len({p["prompt_key"] for p in prompts}) != 1000:
        raise ValueError("Exactly 1000 unique shared prompts required")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    sources = output / "sources"
    sources.mkdir(exist_ok=True)
    attempts = output / "batch_attempts"
    attempts.mkdir(exist_ok=True)
    device = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    if total - free > 512 * 1024**2:
        raise RuntimeError("GPU is occupied; refusing to disturb another workload")
    budget = min(int(total * .90), free - 1024**3)
    lock = {
        "N": 1000, "expected_rows": 3000, "phase": "MAIN_NATIVE_GPU_ONLY",
        "payload": args.payload, "prompt_manifest_sha256": sha(args.manifest),
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0" if args.payload == "xl" else "CompVis/stable-diffusion-v-1-4-original",
        "model_revision": "462165984030d82259a11f4367a4eed129e94a7b" if args.payload == "xl" else "f0bb45b49990512c454cf2c5670b0952ef2f9c71",
        "source_sha256": sha(Path(__file__)), "seed_per_prompt": "20261026 + index",
        "generation": {"device": "cuda", "precision": "XL fp16 with official VAE upcast; V1 official autocast",
                       "steps": 30 if args.payload == "xl" else 50,
                       "guidance": 5. if args.payload == "xl" else 7.5,
                       "resolution": [1024, 1024] if args.payload == "xl" else [512, 512]},
        "batch_policy": {"initial": 2 if args.payload == "xl" else 4,
                         "ceiling": 64,
                         "budget_bytes": budget, "target_fraction": .90,
                         "decision_basis": "Only production-batch memory, never decoded watermark outcomes.",
                         "OOM": "Record failed batch; halve batch and freeze; retry identical indices/seeds.",
                         "last_batch": "May be smaller. Batching may change floating-point outputs; record every batch."},
        "GPU": device.name, "VRAM_bytes": total, "torch": torch.__version__,
        "CUDA": torch.version.cuda, "CPU_data": "Retired and excluded; new GPU-only cohort begins at index 0.",
        "prior_exposure": "Same prompt groups previously partially generated on CPU; chronology disclosed, no outcome-based selection.",
        "no_optional_stopping": True, "margin": .1, "alpha_one_sided": .025,
    }
    frozen_path = output / "prelabel_lock.json"
    records = []
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        authorization = json.loads((output / "resume_authorization.json").read_text())
        if (authorization["old_source_sha256"] != frozen["source_sha256"]
                or authorization["new_source_sha256"] != lock["source_sha256"]):
            raise RuntimeError("GPU source update not authorized")
        for key in ["N", "payload", "prompt_manifest_sha256", "model_id", "model_revision", "generation"]:
            if frozen[key] != lock[key]:
                raise RuntimeError(f"Scientific GPU protocol differs: {key}")
        if authorization["new_batch_ceiling"] != lock["batch_policy"]["ceiling"]:
            raise RuntimeError("Batch-ceiling change not authorized")
        locked_record(output / authorization["execution_lock"], lock)
        for i in range(1000):
            folder = sources / f"{i:04d}"
            done = folder / "completed.json"
            if not done.exists():
                if folder.exists():
                    raise RuntimeError("Uncommitted GPU attempt must be explicitly archived")
                break
            record = json.loads(done.read_text())
            if record["index"] != i or record["prompt_key"] != prompts[i]["prompt_key"]:
                raise RuntimeError("GPU resume prompt order mismatch")
            for name, digest in record["files"].items():
                if sha(folder / name) != digest:
                    raise RuntimeError("GPU resume scientific evidence hash mismatch")
            records.append(record)
        if len(list(sources.glob("*/completed.json"))) != len(records):
            raise RuntimeError("GPU completed source sequence contains gaps")
        print(f"[GPU RESUME VERIFIED] {len(records)} committed sources unchanged", flush=True)
    else:
        save(frozen_path, lock)
    generate = load_xl(args.base, output) if args.payload == "xl" else load_v1(args.base, args.repo, output)
    decoder = WatermarkDecoder("bits", len(PAYLOADS[args.payload]))
    batch = records[-1]["batch_size"] if records else lock["batch_policy"]["initial"]
    ceiling = lock["batch_policy"]["ceiling"]
    frozen_batch = False
    index = len(records)
    attempt_id = max((int(p.name) for p in attempts.iterdir() if p.is_dir()), default=-1) + 1
    while index < 1000:
        indices = list(range(index, min(index + batch, 1000)))
        attempt = attempts / f"{attempt_id:05d}"
        attempt.mkdir()
        save(attempt / "intent.json", {"indices": indices, "seeds": [20261026+i for i in indices],
                                      "requested_batch": batch, "started_unix": time.time()})
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        started = time.monotonic()
        try:
            results = generate([prompts[i]["prompt"] for i in indices], indices)
        except torch.cuda.OutOfMemoryError as error:
            save(attempt / "failure.json", {"type": type(error).__name__, "error": str(error),
                                          "traceback": traceback.format_exc(), "indices": indices,
                                          "policy": "No files committed; same indices/seeds retried at smaller batch."})
            print(f"[OOM RECORDED] {args.payload} batch={batch}; retrying same indices with half batch", flush=True)
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            frozen_batch = True
            attempt_id += 1
            continue
        except Exception as error:
            save(attempt / "failure.json", {"type": type(error).__name__, "error": str(error),
                                          "traceback": traceback.format_exc()})
            raise
        if len(results) != len(indices):
            raise RuntimeError("Generator returned wrong batch size")
        peak = torch.cuda.max_memory_reserved()
        telemetry = {"indices": indices, "actual_batch": len(indices), "peak_reserved_bytes": peak,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "elapsed_seconds": time.monotonic()-started, "budget_bytes": budget}
        save(attempt / "resources.json", telemetry)
        for i, (before, after, issued, flagged) in zip(indices, results):
            folder = sources / f"{i:04d}"
            folder.mkdir()
            np.savez_compressed(folder / "native_boundaries.npz", before=before, after=after)
            issued.save(folder / "issued.png")
            rgb = np.asarray(issued)
            if args.payload == "xl":
                replay_rgb = np.clip(np.rint((after[0].transpose(1, 2, 0) / 2 + .5) * 255), 0, 255).astype(np.uint8)
                if not np.array_equal(rgb, replay_rgb):
                    raise RuntimeError("XL saved boundary-to-image replay differs")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rows = []
            for condition, stages in [
                ("C0", []), ("C444", [{"op": "jpeg", "quality": 95, "subsampling": 0}]),
                ("C420", [{"op": "jpeg", "quality": 95, "subsampling": 2}]),
            ]:
                image = channel(bgr, stages, str(i))
                bits = np.asarray(decoder.decode(image, "dwtDct"), dtype=np.uint8)
                reference = decode(image, len(bits), 36.)
                if reference["status"] != "ok" or not np.array_equal(bits, reference["bits"]):
                    raise RuntimeError("GPU native verifier disagrees with research decoder")
                k = int((bits != PAYLOADS[args.payload]).sum())
                rows.append({"prompt_key": prompts[i]["prompt_key"], "index": i, "seed": 20261026+i,
                             "payload": args.payload, "role": "G", "condition": condition,
                             "K": k, "BER": k/len(bits), "exact": int(k == 0 and not flagged),
                             "tau_accept": int(k <= len(bits)//8 and not flagged),
                             "safety_replaced": flagged, "decoder_exact": int(k == 0),
                             "decoder_tau_accept": int(k <= len(bits)//8)})
            save(folder / "rows.json", rows)
            save(folder / "prompt.json", prompts[i])
            record = {"index": i, "prompt_key": prompts[i]["prompt_key"], "seed": 20261026+i,
                      "native_replay_equal": True, "final_image_replay_equal": True,
                      "safety_replaced": flagged, "device": "cuda", "batch_attempt": attempt_id,
                      "batch_size": len(indices), "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                      "phash": str(perceptual_hash(bgr)),
                      "files": {p.name: sha(p) for p in folder.iterdir() if p.is_file()}}
            save(folder / "completed.json", record)
            records.append(record)
        index += len(indices)
        print(f"[COMMITTED GPU] {args.payload} {index}/1000 batch={len(indices)} peak={peak/1024**3:.2f}GiB replay PASS", flush=True)
        if not frozen_batch and len(indices) == batch:
            selected = next_batch(batch, peak, baseline, budget, ceiling)
            frozen_batch = selected <= batch
            batch = selected
        attempt_id += 1
    duplicates = Counter(r["pixel_sha256"] for r in records)
    save(output / "duplicate_audit.json", {"exact_duplicate_groups": [h for h, n in duplicates.items() if n > 1],
                                          "final_full_pairwise_audit": "PENDING"})
    save(output / "cohort_complete.json", {"status": "GPU_COHORT_COMPLETE_AUDIT_PENDING", "N": 1000, "rows": 3000})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", choices=["xl", "v1"], required=True)
    for name in ["base", "manifest", "output", "repo"]:
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
