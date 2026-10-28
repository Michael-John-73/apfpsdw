"""Strict artifacts, reproducible identifiers, and visible terminal progress."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
import numpy as np
from tqdm import tqdm

class Blocked(RuntimeError):
    """Missing evidence/data; distinct from scientific hypothesis non-support."""

def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Nonfinite values cannot be saved as scientific observations")
    return value

def canonical(value):
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)

def append_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(canonical(row).decode("utf-8") + "\n")
        f.flush()

def read_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical(row).decode("utf-8") + "\n")

def now():
    return datetime.now(timezone.utc).isoformat()

def seed(*parts):
    return int(digest(parts)[:16], 16) % (2**32)

def progress(items, description, total=None):
    return tqdm(items, total=total, desc=description, file=sys.stdout, mininterval=.5, dynamic_ncols=True)

class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log
    def write(self, s):
        self.terminal.write(s)
        self.log.write(s)
        self.log.flush()
        return len(s)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def isatty(self):
        return False
    @property
    def encoding(self):
        return "utf-8"

@contextlib.contextmanager
def terminal_log(root, stage):
    folder = Path(root) / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    original_out, original_err = sys.stdout, sys.stderr
    if hasattr(original_out, "reconfigure"):
        original_out.reconfigure(encoding="utf-8", errors="replace")
    with (folder / f"{stamp}_{stage}.log").open("w", encoding="utf-8") as f:
        sys.stdout, sys.stderr = Tee(original_out, f), Tee(original_err, f)
        start = time.perf_counter()
        try:
            print(f"[START] {stage} UTC={now()}", flush=True)
            yield
            print(f"[DONE] {stage} elapsed_s={time.perf_counter()-start:.3f}", flush=True)
        finally:
            sys.stdout, sys.stderr = original_out, original_err

def source_fingerprint():
    import cv2, pywt, scipy, PIL
    import importlib.metadata as metadata
    root = Path(__file__).parent
    return {"sources": {p.name: file_hash(p) for p in sorted(root.glob("*.py"))},
            "launcher": file_hash(root.parents[2]/"run_paper_b.py"),
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "opencv": cv2.__version__, "pywavelets": pywt.__version__,
                        "scipy": scipy.__version__, "pillow": PIL.__version__,
                        **{name:metadata.version(name) for name in ("torch","diffusers","invisible-watermark","tqdm")}}}

