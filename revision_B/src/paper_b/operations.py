"""Operational security controls and end-to-end cost; not model selection."""
from collections import defaultdict
from pathlib import Path
import csv
import time
import numpy as np
from . import codec
from .common import Blocked,read_json,write_json,read_rows,save_rows,progress,digest,file_hash,seed
from .probability import paired_interval,exact_ci
from .models import predict_pmf,predict_pmfs

def import_legacy(s):
    root=Path(s.c["paths"]["legacy_root"]);summaries=[];artifacts=[]
    for domain,sub in (("real","s5_e2"),("ai","s7_e4")):
        folder=root/"outputs_conf"/sub/"shards"
        for csvpath in progress(sorted(folder.glob("*.csv")),"import ICISC "+domain):
            npypath=csvpath.with_suffix(".npy")
            if not npypath.is_file():continue
            entries=list(csv.DictReader(csvpath.open(encoding="utf-8-sig")))
            matrix=np.load(npypath,allow_pickle=False)
            payload="xl" if "xl_" in csvpath.stem else "v1" if "v1_" in csvpath.stem else None
            if payload is None:continue
            ref=codec.PAYLOADS[payload]
            valid=[r for r in entries if int(r["decodable"])==1]
            if matrix.shape!=(len(valid),len(ref)) or not np.isin(matrix,[0,1]).all():raise ValueError("Legacy CSV/NPY alignment invalid: "+str(csvpath))
            restored=matrix.astype(np.uint8)^ref
            if not np.array_equal((restored!=ref).astype(np.uint8),matrix):raise ValueError("Legacy bit restoration mismatch")
            # Only compare historical CSV metrics that are actually present.
            for r,errors in zip(valid,matrix):
                if "ber" in r and abs(float(r["ber"])-float(errors.mean()))>5.1e-7:raise ValueError("Legacy BER mismatch")
            tag=domain+"_"+csvpath.stem
            output=s.root/"legacy_errors"/(tag+".npz");output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output,errors=matrix.astype(np.uint8),decoded_bits=restored)
            artifacts.append(str(output.relative_to(s.root)))
            summaries.append({"domain":domain,"file":str(csvpath),"csv_sha256":file_hash(csvpath),"npy_sha256":file_hash(npypath),
                "n":len(entries),"decodable":len(valid),"guard_or_decode_failure":len(entries)-len(valid),
                "payload":payload,"exact":exact_ci(int(np.sum(matrix.sum(axis=1)==0)),len(entries)),
                "tau":exact_ci(int(np.sum(matrix.sum(axis=1)<=int(.125*len(ref)))),len(entries)),
                "mean_ber_conditional":float(matrix.mean()) if len(valid) else None,
                "scope":"historical development evidence; not fresh confirmation"})
    if not summaries:raise Blocked("No compatible ICISC CSV/NPY pairs")
    write_json(s.root/"legacy_import.json",summaries)
    s.finish("import-legacy",["legacy_import.json"]+artifacts)

def security(s):
    s.require("freeze","evaluate")
    sources=[r for r in read_json(s.root/"data_manifest.json") if r["split"]=="test"]
    if len(sources)%2:sources=sources[:-1]
    rows=[];length=48;delta=s.c["measurement"]["delta"]
    fixed_secret=np.random.default_rng(seed(s.c["statistics"]["seed"],"one_fixed_secret")).integers(0,2,length,dtype=np.uint8)
    # Pair-disjoint reference and target groups; no shared donor induces fake independence.
    for i in progress(range(0,len(sources),2),"secret and reference scenario audit"):
        donor,target=sources[i:i+2];reference=s._verify_image(donor);image=s._verify_image(target)
        rng=np.random.default_rng(seed(s.c["statistics"]["seed"],donor["group_id"],target["group_id"],"secret_control"))
        for repeat in range(4):
            secret=rng.integers(0,2,length,dtype=np.uint8)
            guess=rng.integers(0,2,length,dtype=np.uint8)
            refmark=codec.encode_fast(reference,secret,delta,"uint8",True)
            extracted=codec.decode(refmark,length,delta)["bits"]
            fixed_extracted=codec.decode(codec.encode_fast(reference,fixed_secret,delta,"uint8",True),length,delta)["bits"]
            scenarios=[("legitimate_known_secret",secret,secret), ("blind_one_guess_no_reference",guess,secret),
                       ("one_reference_decode_reinsert",extracted,secret),("public_known_fixed_payload",codec.PAYLOADS["xl"],codec.PAYLOADS["xl"])]
            if repeat==0:
                scenarios += [("fixed_secret_legitimate",fixed_secret,fixed_secret),("fixed_secret_blind_guess",guess,fixed_secret),
                              ("fixed_secret_one_reference",fixed_extracted,fixed_secret)]
            for cond in s.c["measurement"]["conditions"]:
                for scenario,payload,expected in scenarios:
                    marked=codec.encode_fast(image,payload,delta,"uint8",True)
                    out=codec.channel(marked,cond["stages"],Path(target["path"]).name)
                    decoded=codec.decode(out,length,delta)
                    error=None if decoded["bits"] is None else int(np.sum(decoded["bits"]!=expected))
                    rows.append({"pair":digest([donor["group_id"],target["group_id"]]),"reference":donor["image_id"],"target":target["image_id"],
                        "repeat":repeat,"scenario":scenario,"condition":cond["id"],"K":error,"exact":int(error==0),
                        "tau_accept":int(error is not None and error<=6),"reference_exact":int(np.array_equal(fixed_extracted,fixed_secret) if scenario.startswith("fixed_secret") else np.array_equal(extracted,secret)),
                        "attempt_budget":1,"marking_kind":"uint8","swap":True,"scope":"controlled bit-secret experiment; not a cryptographic construction"})
    summary=[]
    for scenario,cond in sorted({(r["scenario"],r["condition"]) for r in rows}):
        selected=[r for r in rows if r["scenario"]==scenario and r["condition"]==cond];groups=defaultdict(list)
        for r in selected:groups[r["pair"]].append(r["exact"])
        ci=paired_interval([np.mean(v) for v in groups.values()],s.c["statistics"]["seed"],s.c["statistics"]["bootstrap_reps"],s.c["statistics"]["confidence"])
        summary.append({"scenario":scenario,"condition":cond,"all_attempts":len(selected),"independent_pairs":len(groups),
                        "exact_pair_cluster_interval":ci,"tau_all_attempt_rate":float(np.mean([r["tau_accept"] for r in selected])),
                        "reference_correct_rate":float(np.mean([r["reference_exact"] for r in selected]))})
    save_rows(s.root/"security_rows.jsonl",rows)
    write_json(s.root/"security.json",{"summary":summary,"threat_contract":"blind sees neither secret nor marked reference; reference scenario gets one marked donor and the public decoder; public payload is known",
        "limitations":"We use four independent-secret repeats per pair and one fixed secret across the corpus, with one attempt/pair in fixed-secret scenarios. We condition fixed-secret intervals on that secret rather than infer a population-of-secrets effect. We report zero-event counts without inferring zero risk from degenerate bootstrap intervals.",
        "ideal_independent_secret_chance":{"exact":2**-48,"tau":sum(__import__('math').comb(48,k) for k in range(7))/2**48}})
    s.finish("security",["security_rows.jsonl","security.json"])

def cost(s):
    s.require("select","measure-pilot")
    models=read_json(s.root/("calibrated_models.json" if (s.root/"calibrated_models.json").is_file() else "selected_models.json"))
    sources=[r for r in read_json(s.root/"data_manifest.json") if r["split"]=="pilot"]
    rows=[]
    # Warming avoids charging library first-use setup to only one method.
    warm=s._verify_image(sources[0]);codec.decode(codec.encode_fast(warm,codec.PAYLOADS["xl"]),48)
    for source in progress(sources,"full prediction versus exact replay cost"):
        for variant in s.c["measurement"]["variants"]:
            for payload in s.c["measurement"]["payloads"]:
                for cond in s.c["measurement"]["conditions"]:
                    # Alternate order per source, retaining unamortized full costs.
                    durations={}
                    for mode in (["predict","replay"] if int(source["image_id"],16)%2 else ["replay","predict"]):
                        start=time.perf_counter();image=s._verify_image(source)
                        if mode=="predict":
                            _,trace=codec.encode(image,codec.PAYLOADS[payload],s.c["measurement"]["delta"],variant["kind"],variant["swap"])
                            f=codec.feature_dict(trace,variant,len(codec.PAYLOADS[payload]),s.c["measurement"]["delta"],cond["stages"])
                            row={"L":len(codec.PAYLOADS[payload]),"variant":variant["id"],"payload_id":payload,"condition_id":cond["id"],"features":f}
                            predict_pmfs(models["proposed"],[row],s.c.get("compute"))
                        else:
                            marked=codec.encode_fast(image,codec.PAYLOADS[payload],s.c["measurement"]["delta"],variant["kind"],variant["swap"])
                            out=codec.channel(marked,cond["stages"],Path(source["path"]).name)
                            codec.decode(out,len(codec.PAYLOADS[payload]),s.c["measurement"]["delta"])
                        durations[mode]=time.perf_counter()-start
                    rows.append({"image_id":source["image_id"],"variant":variant["id"],"payload":payload,"condition":cond["id"],**durations})
    summary={}
    for key in ("predict","replay"):
        values=[r[key] for r in rows];summary[key]={"median_s":float(np.median(values)),"p95_s":float(np.quantile(values,.95)),"sum_s":float(np.sum(values))}
    save_rows(s.root/"cost_rows.jsonl",rows)
    write_json(s.root/"cost.json",{"summary":summary,"includes":"image verification/read, full encode, features+PMF or channel+decode; both paths include source I/O; warm OS cache; unamortized per query",
        "excludes":"common original-image generation, one-time fitting/calibration (recorded separately), network download and serialization of final tables",
        "interpretation":"We measure single-query latency including backend policy and transfer overhead, and record large-batch numerical throughput separately in gpu_profile.json. We compare prediction with exact replay; kernel throughput alone does not establish end-to-end acceleration."})
    s.finish("cost",["cost_rows.jsonl","cost.json"])

def mechanisms(s):
    available=[split for split in ("pilot","fit","test") if s.state["stages"].get("measure-"+split,{}).get("status")=="TECH_PASS"]
    if not available:raise Blocked("Mechanism analysis needs committed measurements")
    split=available[-1];rows=[r for r in read_rows(s.root/(split+"_rows.jsonl")) if r["role"]=="public_mark"]
    cells=defaultdict(dict)
    for r in rows:cells[(r["group_id"],r["payload_id"],r["condition_id"])][(r["kind"],r["swap"])]=r
    effects=defaultdict(lambda:defaultdict(list))
    for (group,payload,condition),cell in cells.items():
        needed={(k,v) for k in ("uint8","float32") for v in (True,False)}
        if not needed<=set(cell):raise ValueError("Mechanism factorial design incomplete")
        for metric in ("ber","mse_rgb","exact"):
            def value(r):return None if metric=="ber" and r["K"] is None else (r["K"]/r["L"] if metric=="ber" else r[metric])
            v={k:value(r) for k,r in cell.items()}
            if any(x is None for x in v.values()):continue
            contrast=(v[("float32",True)]-v[("float32",False)])-(v[("uint8",True)]-v[("uint8",False)])
            effects[(payload,condition,metric)][group].append(contrast)
    results=[]
    for key,groups in effects.items():
        values=[np.mean(v) for v in groups.values()]
        if len(values)<2:continue
        ci=paired_interval(values,s.c["statistics"]["seed"],s.c["statistics"]["bootstrap_reps"],s.c["statistics"]["confidence"])
        results.append({"payload":key[0],"condition":key[1],"metric":key[2],"dtype_by_swap_interaction":ci})
    write_json(s.root/"mechanisms.json",{"split":split,"results":results,"interpretation":"Controlled component interventions; descriptive multiple contrasts, no forced causal direction or adjusted confirmatory mechanism claim. BER excludes guard refusals; exact retains them."})
    s.finish("mechanisms",["mechanisms.json"])
