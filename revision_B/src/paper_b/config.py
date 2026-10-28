"""Validated configuration; unresolved scientific criteria block confirmation only."""
from pathlib import Path
import re
import numpy as np
from .common import read_json

SCHEMA={
"study_id":None,
"paths":{"legacy_root":None,"raw_root":None,"work_root":None,"run_root":None},
"data":{"fresh_sources":None,"pilot_per_domain":None,"validation_fraction":None,"min_side":None,"phash_distance":None,"development_limit_per_domain":None},
"measurement":{"delta":None,"payloads":None,"variants":None,"conditions":None,"roles":None},
"training":{"l2_candidates":None,"maxiter":None,"ftol":None,"gtol":None,"seed":None},
"calibration":{"temperature_bounds":None,"maxiter":None},
"statistics":{"confidence":None,"bootstrap_reps":None,"seed":None},
"design":{"delta_nll":None,"delta_justification":None,"acceptance_mae":None,
          "mae_justification":None,"power":None,"proportion_half_width":None,
          "max_test_groups":None,"calibration_groups":None,"test_groups":None,
          "source_scope":None,"require_source_proof":None},
"environment":{"profile":None,"require_cuda":None,"expected_gpu":None,"cpu_threads":None}
}
def _keys(value,schema,path="config"):
    if not isinstance(value,dict) or set(value)!=set(schema):
        missing=set(schema)-set(value) if isinstance(value,dict) else set(schema)
        extra=set(value)-set(schema) if isinstance(value,dict) else set()
        raise ValueError(f"{path}: missing={sorted(missing)} unknown={sorted(extra)}")
    for key,sub in schema.items():
        if sub is not None:_keys(value[key],sub,path+"."+key)

def load(path):
    c=read_json(path)
    from .batch_backend import DEFAULTS,settings
    schema=dict(SCHEMA)
    if "compute" in c:schema["compute"]={k:None for k in DEFAULTS}
    _keys(c,schema)
    if "compute" in c:settings(c["compute"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+",c["study_id"]):
        raise ValueError("Invalid study_id")
    for key,value in c["paths"].items():
        if not isinstance(value,str) or not value or "TODO" in value:
            raise ValueError("Unresolved path "+key)
        if c["environment"]["profile"]=="runpod" and not value.replace("\\","/").startswith("/workspace/"):
            raise ValueError("RunPod data/results must be under /workspace: "+key)
    if not 0<c["data"]["validation_fraction"]<.5 or c["data"]["pilot_per_domain"]<1:
        raise ValueError("Invalid development split")
    if c["data"]["min_side"]<256 or not 0<=c["data"]["phash_distance"]<=8:
        raise ValueError("Invalid prepared-image or duplicate policy")
    for src in c["data"]["fresh_sources"]:
        if set(src)!={"domain","path","metadata"} or src["domain"] not in ("real","ai"):
            raise ValueError("fresh source schema is domain/path/metadata")
    m=c["measurement"]
    if not np.isfinite(m["delta"]) or m["delta"]<=0 or not set(m["payloads"])<=set(("xl","v1")) or not m["payloads"]:
        raise ValueError("Invalid delta/payload")
    if m["roles"]!=["public_mark","unmarked"]:
        raise ValueError("Both marked and natural-negative controls are mandatory")
    ids=[]
    for v in m["variants"]:
        if set(v)!={"id","kind","swap"} or v["kind"] not in ("uint8","float32") or type(v["swap"]) is not bool:
            raise ValueError("Invalid variant")
        ids.append(v["id"])
    if len(set(ids))!=len(ids):raise ValueError("Duplicate variant ids")
    cond_ids=[]
    for q in m["conditions"]:
        if set(q)!={"id","panel","stages"} or q["panel"] not in ("primary","transfer_order","transfer_length"):
            raise ValueError("Invalid condition")
        cond_ids.append(q["id"])
        for s in q["stages"]:
            expected={"jpeg":{"op","quality","subsampling"},"chroma":{"op","fx","fy"},
                      "resize":{"op","scale"},"crop":{"op","area"},"blur":{"op","sigma"},"noise":{"op","sigma"}}
            if s.get("op") not in expected or set(s)!=expected[s["op"]]:raise ValueError("Unknown channel parameter")
            if s["op"]=="jpeg" and (not 1<=s["quality"]<=100 or s["subsampling"] not in (0,1,2)):raise ValueError("Invalid JPEG")
            if s["op"]=="chroma" and (s["fx"] not in (1,2) or s["fy"] not in (1,2)):raise ValueError("Invalid chroma factor")
            if s["op"]=="resize" and not 0<s["scale"]<=2:raise ValueError("Invalid resize scale")
            if s["op"]=="crop" and not 0<s["area"]<=1:raise ValueError("Invalid retained area")
            if s["op"] in ("noise","blur") and not 0<=s["sigma"]<=5:raise ValueError("Invalid sigma")
    if len(set(cond_ids))!=len(cond_ids) or not any(q["panel"]=="primary" for q in m["conditions"]):
        raise ValueError("Duplicate/empty primary conditions")
    t=c["training"]
    if t["maxiter"]<10 or t["ftol"]<=0 or t["gtol"]<=0 or not t["l2_candidates"] or any(x<0 for x in t["l2_candidates"]):
        raise ValueError("Invalid optimizer settings")
    if len(t["l2_candidates"])>4:raise ValueError("At most four declared regularization candidates")
    b=c["calibration"]["temperature_bounds"]
    if not .25<=b[0]<1<b[1]<=4 or c["calibration"]["maxiter"]<10:raise ValueError("Invalid calibration settings")
    if not .9<c["statistics"]["confidence"]<1 or c["statistics"]["bootstrap_reps"]<100:
        raise ValueError("Invalid confidence/bootstrap")
    if c["environment"]["cpu_threads"]<1:raise ValueError("Invalid CPU threads")
    if c["environment"]["profile"] not in ("local","runpod"):raise ValueError("Unknown environment profile")
    if type(c["environment"]["require_cuda"]) is not bool or type(c["design"]["require_source_proof"]) is not bool:
        raise ValueError("Boolean settings must be true/false")
    integer_fields=[c["data"][k] for k in ("pilot_per_domain","min_side","phash_distance")]+[
        t["maxiter"],t["seed"],c["calibration"]["maxiter"],c["statistics"]["bootstrap_reps"],c["statistics"]["seed"],
        c["environment"]["cpu_threads"],c["design"]["calibration_groups"],c["design"]["max_test_groups"]]
    if any(type(x) is not int or x<0 for x in integer_fields):raise ValueError("Expected nonnegative integer setting")
    limit=c["data"]["development_limit_per_domain"]
    if limit is not None and (type(limit) is not int or limit<c["data"]["pilot_per_domain"]+4):raise ValueError("Invalid development limit")
    if c["environment"]["profile"]=="runpod" and limit is not None:raise ValueError("Confirmatory RunPod config cannot truncate development silently")
    d=c["design"]
    if not .5<d["power"]<1 or not 0<d["proportion_half_width"]<.5 or d["calibration_groups"]<4 or d["max_test_groups"]<4:
        raise ValueError("Invalid sample design")
    if d["test_groups"] is not None and (type(d["test_groups"]) is not int or d["test_groups"]<4):raise ValueError("Invalid test group allocation")
    numeric=[m["delta"],t["ftol"],t["gtol"],*t["l2_candidates"],*b,c["statistics"]["confidence"],d["power"],d["proportion_half_width"]]
    if not np.isfinite(numeric).all():raise ValueError("Nonfinite configuration")
    if len(set(m["payloads"]))!=len(m["payloads"]) or not m["variants"]:raise ValueError("Empty/duplicate measurement design")
    return c

def confirmatory_requirements(config):
    d=config["design"]
    gaps=[]
    for key in ("delta_nll","delta_justification","acceptance_mae","mae_justification","test_groups"):
        if d[key] is None or d[key]=="":gaps.append("design."+key)
    if d["delta_nll"] is not None and (not np.isfinite(d["delta_nll"]) or d["delta_nll"]<=0):gaps.append("positive delta_nll")
    if d["acceptance_mae"] is not None and not 0<d["acceptance_mae"]<1:gaps.append("acceptance_mae in (0,1)")
    if d["test_groups"] is not None and d["test_groups"]>d["max_test_groups"]:gaps.append("sample budget")
    return gaps

