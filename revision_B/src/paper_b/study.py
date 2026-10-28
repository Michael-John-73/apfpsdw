"""Reproducible stage state and per-image atomic measurement checkpoints."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import copy
import csv
import importlib.metadata as metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import cv2
import numpy as np
from . import codec
from .common import (Blocked,append_json,canonical,digest,file_hash,now,progress,
                     read_json,read_rows,save_rows,source_fingerprint,write_json)
from .config import load,confirmatory_requirements
from .data import inventory,split_development,assign_fresh,image_read,prepared
from .models import (BASELINES,MODEL_NAMES,fit_model,score_model,calibrate_model,
                     predict_pmf,usable)
from .probability import acceptance,exact_ci,paired_interval

class Study:
    def __init__(self,config_path,create=False):
        self.config_path=Path(config_path).resolve()
        self.input_config=load(self.config_path)
        self.root=Path(self.input_config["paths"]["run_root"])/self.input_config["study_id"]
        if create:
            if (self.root/"state.json").exists():raise ValueError("Run already initialized; inspect status")
            self.root.mkdir(parents=True,exist_ok=True)
            self.c=copy.deepcopy(self.input_config)
            self.state={"study_id":self.c["study_id"],"created":now(),"input_config_hash":digest(self.input_config),
                        "implementation":source_fingerprint(),"stages":{},"artifacts":{},
                        "test_accessed":False,"hypotheses":{"H_BP":"NOT_EVALUATED","H_BT":"NOT_EVALUATED"},
                        "test_readiness":"NOT_READY","source_scope":"component_watermarker_only"}
            write_json(self.root/"input_config.json",self.input_config)
            write_json(self.root/"resolved_config.json",self.c)
            self.finish("init",["input_config.json","resolved_config.json"])
        else:
            if not (self.root/"state.json").is_file():raise Blocked("Initialize the run first")
            self.state=read_json(self.root/"state.json")
            if digest(self.input_config)!=self.state["input_config_hash"]:raise ValueError("Input config changed; a new run is required")
            if source_fingerprint()!=self.state["implementation"]:raise ValueError("Source/runtime changed since init; preserve the old run and start a new revision")
            self.c=load(self.root/"resolved_config.json")
            for name,h in self.state["artifacts"].items():
                if file_hash(self.root/name)!=h:raise ValueError("Committed artifact changed: "+name)

    def require(self,*stages):
        for stage in stages:
            if self.state["stages"].get(stage,{}).get("status")!="TECH_PASS":
                raise Blocked("Required stage not passed: "+stage)

    def finish(self,stage,artifacts,**extra):
        self.state.update(extra)
        for name in artifacts:self.state["artifacts"][name]=file_hash(self.root/name)
        self.state["stages"][stage]={"status":"TECH_PASS","finished":now(),"artifacts":artifacts}
        write_json(self.root/"state.json",self.state)
        print(f"[TECH_PASS] {stage}; scientific verdicts remain separate",flush=True)

    def failure(self,stage,exc):
        status="BLOCKED" if isinstance(exc,Blocked) else "FAIL"
        self.state["stages"][stage]={"status":status,"finished":now(),"reason":str(exc)}
        write_json(self.root/"state.json",self.state)
        append_json(self.root/"issues.jsonl",{"time":now(),"stage":stage,"type":type(exc).__name__,"message":str(exc)})
        print(f"[{status}] {stage}: {exc}",flush=True)

    def environment(self):
        expected_root=Path(__file__).resolve().parents[3]
        if Path(self.c["paths"]["work_root"]).resolve()!=expected_root:
            raise ValueError("paths.work_root must be the actual launcher/source folder")
        env={"platform":platform.platform(),"python":sys.version,"cpu_count":os.cpu_count(),
             "cpu_threads":self.c["environment"]["cpu_threads"],"core_compute":"CPU reference or bounded CUDA float64 objective/PMFs; SciPy L-BFGS-B controller on CPU",
             "compute_requested":self.c.get("compute",{"device":"cpu","reason":"legacy config"}),
             "versions":{},"mounts":{}}
        for name in ("numpy","scipy","PyWavelets","Pillow","opencv-python","tqdm","torch","diffusers","invisible-watermark"):
            try:env["versions"][name]=metadata.version(name)
            except metadata.PackageNotFoundError:env["versions"][name]=None
        for name,p in self.c["paths"].items():
            target=Path(p)
            while not target.exists() and target!=target.parent:target=target.parent
            usage=shutil.disk_usage(target)
            env["mounts"][name]={"configured":p,"existing_parent":str(target),"free_bytes":usage.free,
                                  "mountpoint":os.path.ismount(target)}
        try:
            import torch
            env["cuda_available"]=torch.cuda.is_available()
            if torch.cuda.is_available():
                x=torch.ones(32,device="cuda");y=x@x
                torch.cuda.synchronize()
                env["gpu"]={"name":torch.cuda.get_device_name(0),"count":torch.cuda.device_count(),
                            "total_bytes":torch.cuda.get_device_properties(0).total_memory,
                            "free_total":list(torch.cuda.mem_get_info()),"smoke_dot":float(y.cpu())}
            else:env["gpu"]=None
        except ImportError:
            env["cuda_available"]=False;env["gpu"]=None
        write_json(self.root/"environment.json",env)
        if self.c["environment"]["profile"]=="runpod":
            try:
                mount=subprocess.run(["findmnt","--json","--target","/workspace"],capture_output=True,text=True,check=True)
                env["workspace_mount"]=json.loads(mount.stdout)
                env["storage_persistence"]="Record mount source; operator must retain the volume. A pathname alone does not prove persistence."
            except (OSError,subprocess.CalledProcessError,ValueError) as exc:
                raise Blocked("Cannot verify the /workspace backing filesystem: "+str(exc))
            for key in ("work_root","raw_root","run_root"):
                path=Path(self.c["paths"][key]);path.mkdir(parents=True,exist_ok=True)
                probe=path/".paper_b_write_probe"
                probe.write_text("write verified",encoding="utf-8");probe.unlink()
            write_json(self.root/"environment.json",env)
        if self.c["environment"]["require_cuda"]:
            if not env["cuda_available"]:raise Blocked("RunPod CUDA environment not available; local CPU diagnostics do not satisfy the GPU environment gate")
            if self.c["environment"]["expected_gpu"] not in env["gpu"]["name"]:raise Blocked("Observed GPU differs from planned GPU")
            if "4090" in env["gpu"]["name"] and env["gpu"]["total_bytes"]<22*1024**3:raise Blocked("4090 reported less memory than the 24GB resource plan")
        self.finish("environment",["environment.json"])

    def source_audit(self):
        import imwatermark
        from imwatermark import WatermarkEncoder
        import imwatermark.maxDct as legacy
        report={"historical_commit_reproduced":False,"scope":"installed watermarker component equivalence, not full generator issuance",
                "library_file":inspect.getfile(legacy),"library_sha256":file_hash(inspect.getfile(legacy)),"checks":[]}
        rng=np.random.default_rng(303)
        images=[rng.integers(0,256,(256,256,3),dtype=np.uint8),
                rng.integers(0,256,(320,384,3),dtype=np.uint8)]
        for key,bits in codec.PAYLOADS.items():
            enc=WatermarkEncoder();enc.set_watermark("bits",bits.tolist())
            for i,image in enumerate(progress(images,"installed uint8 parity "+key)):
                expected=enc.encode(image,"dwtDct")
                got,_=codec.encode(image,bits,kind="uint8",swap=True)
                report["checks"].append({"path":"library_uint8","payload":key,"case":i,"pixel_equal":bool(np.array_equal(expected,got))})
        try:
            import torch
            from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker
            source_file=inspect.getfile(StableDiffusionXLWatermarker)
            report["sdxl_source_file"]=source_file
            report["sdxl_source_sha256"]=file_hash(source_file)
            marker=StableDiffusionXLWatermarker()
            for i,image in enumerate(progress(images,"installed SDXL component parity")):
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255
                incoming=torch.from_numpy(rgb).permute(2,0,1)[None]*2-1
                returned=marker.apply_watermark(incoming)
                a=np.clip(np.rint((returned[0].permute(1,2,0).numpy()+1)/2*255),0,255).astype(np.uint8)
                expected=cv2.cvtColor(a,cv2.COLOR_RGB2BGR)
                got,_=codec.encode(image,codec.PAYLOADS["xl"],kind="float32",swap=True)
                report["checks"].append({"path":"native_sdxl_48_component","case":i,"pixel_equal":bool(np.array_equal(expected,got)),
                                        "max_pixel_difference":int(np.abs(expected.astype(int)-got.astype(int)).max())})
        except ImportError as exc:report["sdxl_unavailable"]=str(exc)
        report["all_equal"]=all(r["pixel_equal"] for r in report["checks"])
        report["sdxl_verified"]=sum(r["path"]=="native_sdxl_48_component" for r in report["checks"])==len(images)
        write_json(self.root/"source_audit.json",report)
        if not report["all_equal"]:raise ValueError("Installed component parity failed")
        if self.c["design"]["require_source_proof"] and not report["sdxl_verified"]:raise Blocked("SDXL reference component not verified")
        self.finish("source-audit",["source_audit.json"])

    def gpu_profile(self):
        self.require("environment")
        from .gpu_checks import cuda_profile
        from .batch_backend import settings
        report=cuda_profile(settings(self.c.get("compute")))
        write_json(self.root/"gpu_profile.json",report)
        if report["status"]!="PASS_CUDA":raise Blocked("No actual CUDA device; CPU math checks passed but 4090 VRAM/throughput remains NOT_RUN_CUDA")
        self.finish("gpu-profile",["gpu_profile.json"])

    def prepare(self):
        self.require("environment")
        if "prepare" in self.state["stages"] and self.state["stages"]["prepare"]["status"]=="TECH_PASS":
            raise ValueError("Data manifest already committed; do not change the dataset under this run")
        rows,pairs,all_rows=inventory(self.c)
        manifest=split_development(rows,self.c)
        write_json(self.root/"data_manifest.json",manifest)
        write_json(self.root/"data_audit.json",{"all_files":len(all_rows),"independent_representatives":len(rows),
            "near_duplicate_pairs":pairs,"split_counts":dict(Counter(r["split"] for r in manifest)),
            "legacy_ai_group_evidence":dict(Counter(r["group_evidence"] for r in all_rows if r["historical"] and r["domain"]=="ai")),
            "phash_policy":"conservative duplicate heuristic; not a proof that all semantic duplicates are absent"})
        self.finish("prepare",["data_manifest.json","data_audit.json"])

    def _verify_image(self,row):
        if file_hash(row["path"])!=row["sha256"]:raise ValueError("Image changed since dataset commitment: "+row["image_id"])
        return prepared(image_read(row["path"]),self.c["data"]["min_side"])

    def _measure_image(self,source):
        image=self._verify_image(source)
        rows=[]
        for variant in self.c["measurement"]["variants"]:
            for payload_id in self.c["measurement"]["payloads"]:
                bits=codec.PAYLOADS[payload_id]
                start=time.perf_counter()
                marked,trace=codec.encode(image,bits,self.c["measurement"]["delta"],variant["kind"],variant["swap"])
                encode_time=time.perf_counter()-start
                for condition in self.c["measurement"]["conditions"]:
                    if source["split"] in ("pilot","fit","validation","calibration") and condition["panel"]!="primary":continue
                    features=codec.feature_dict(trace,variant,len(bits),self.c["measurement"]["delta"],condition["stages"])
                    h,w=codec.planned_geometry(*image.shape[:2],condition["stages"])
                    for role in self.c["measurement"]["roles"]:
                        start=time.perf_counter()
                        transformed=codec.channel(marked if role=="public_mark" else image,condition["stages"],Path(source["path"]).name)
                        result=codec.decode(transformed,len(bits),self.c["measurement"]["delta"])
                        support=h*w>=65536
                        if transformed.shape[:2]!=(h,w):raise ValueError("Planned and observed channel geometry disagree")
                        row={k:source[k] for k in ("image_id","group_id","domain","split")}
                        trial={"image":source["image_id"],"variant":variant["id"],"payload":payload_id,
                               "condition":condition["id"],"role":role}
                        row.update(trial_id=digest(trial)[:32],variant=variant["id"],kind=variant["kind"],swap=variant["swap"],
                           payload_id=payload_id,L=len(bits),condition_id=condition["id"],panel=condition["panel"],
                           role=role,status=result["status"],decoder_supported=support,
                           source_hash=source["sha256"],height=image.shape[0],width=image.shape[1],
                           output_height=h,output_width=w,features=features,K=None,errors=None,bits=None,
                           exact=0,tau_accept=0,mse_rgb=trace["mse_rgb"],psnr=trace["psnr"],
                           psnr_infinite=trace["psnr_infinite"],encode_and_features_s=encode_time,
                           channel_and_decode_s=time.perf_counter()-start,
                           source_scope="watermarker_component_surrogate")
                        if result["status"]=="ok":
                            errors=(result["bits"]!=bits).astype(np.uint8)
                            row.update(K=int(errors.sum()),errors=errors.tolist(),bits=result["bits"].tolist(),
                                       exact=int(not errors.any()),tau_accept=int(errors.sum()<=int(.125*len(bits))))
                            geometric=any(s["op"] in ("resize","crop") for s in condition["stages"])
                            if role=="public_mark" and not geometric and len(result["block_pos"])==len(trace["pre_block_pos"]):
                                row["postchannel_argmax_same"]=float(np.mean(result["block_pos"]==trace["pre_block_pos"]))
                                row["postchannel_block_error"]=float(np.mean(result["block_bits"]!=trace["embedded_block_bits"]))
                        elif support:
                            raise ValueError("Decoder rejected a geometrically supported image")
                        rows.append(row)
        return rows

    def measure(self,split):
        self.require("prepare","source-audit","mathcheck")
        if split not in ("pilot","fit","validation","calibration","test"):raise ValueError("Unsupported split")
        if split=="calibration":self.require("select","plan")
        if split=="test":self.require("freeze","forecast")
        if split in ("pilot","fit","validation") and self.state["test_accessed"]:raise Blocked("Development after test access requires a new study and fresh test")
        manifest=read_json(self.root/"data_manifest.json")
        sources=[r for r in manifest if r["split"]==split]
        if not sources:raise Blocked("No committed sources for split "+split)
        if split=="test":
            self.state["test_accessed"]=True
            write_json(self.root/"state.json",self.state)
        append_json(self.root/"data_access.jsonl",{"time":now(),"split":split,"access":"generate_measurement_labels","groups":len(sources)})
        folder=self.root/"measurements"/split
        folder.mkdir(parents=True,exist_ok=True)
        commitment=digest({"measurement":self.c["measurement"],"min_side":self.c["data"]["min_side"],
                           "implementation":self.state["implementation"]})
        for source in sources:
            p=folder/(source["image_id"]+".json")
            if p.exists():
                cached=read_json(p)
                if cached["commitment"]!=commitment or cached["source_hash"]!=source["sha256"] or digest(cached["rows"])!=cached["rows_hash"]:
                    raise ValueError("Invalid measurement checkpoint: "+str(p))
        pending=[s for s in sources if not (folder/(s["image_id"]+".json")).is_file()]
        print(f"[MEASURE] split={split} groups={len(sources)} cached={len(sources)-len(pending)} pending={len(pending)}",flush=True)
        # Bounded futures prevent accumulating an entire image/condition bank in RAM.
        workers=self.c["environment"]["cpu_threads"]
        cv2.setNumThreads(1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            queue=iter(pending);active={}
            for _ in range(min(workers,len(pending))):
                src=next(queue);active[pool.submit(self._measure_image,src)]=src
            bar=progress(range(len(pending)),"measure "+split)
            while active:
                ready,_=wait(active,return_when=FIRST_COMPLETED)
                for future in ready:
                    source=active.pop(future)
                    rows=future.result()
                    write_json(folder/(source["image_id"]+".json"),{"commitment":commitment,"source_hash":source["sha256"],
                                                                "rows":rows,"rows_hash":digest(rows)})
                    bar.update(1)
                    try:
                        src=next(queue);active[pool.submit(self._measure_image,src)]=src
                    except StopIteration:
                        continue
            bar.close()
        rows=[]
        for source in progress(sources,"aggregate "+split):
            rows.extend(read_json(folder/(source["image_id"]+".json"))["rows"])
        ids=[r["trial_id"] for r in rows]
        if len(ids)!=len(set(ids)):raise ValueError("Duplicate trials")
        path=f"{split}_rows.jsonl";save_rows(self.root/path,rows)
        self.finish("measure-"+split,[path])

