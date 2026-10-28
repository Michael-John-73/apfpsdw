"""Dataset inventory, duplicate grouping, and development/fresh-data separation."""
from __future__ import annotations
import csv
from pathlib import Path
import json
from collections import defaultdict
import cv2
import numpy as np
from .common import Blocked,digest,file_hash,progress,seed

def image_read(path):
    data=Path(path).read_bytes()
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError("Unreadable image: "+str(path))
    return image

def perceptual_hash(image):
    small=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    dct=cv2.dct(small)[:8,:8].ravel()
    bits=dct[1:]>np.median(dct[1:])
    return sum(int(b)<<i for i,b in enumerate(bits))

def prepared(image,min_side):
    h,w=image.shape[:2]
    if min(h,w)<min_side:
        ratio=min_side/min(h,w)
        return cv2.resize(image,(round(w*ratio),round(h*ratio)),interpolation=cv2.INTER_CUBIC)
    return image

def _metadata(path):
    if not path:return {}
    p=Path(path)
    if p.is_dir():
        result={}
        for item in sorted(p.glob("*.json")):
            val=json.loads(item.read_text(encoding="utf-8"))
            if isinstance(val,dict):result.update(val)
        return result
    val=json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(val,dict):raise ValueError("Generated-image metadata must be a keyed object")
    return val

def _prompt_key(meta,filename):
    record=meta.get(filename,meta.get(Path(filename).stem))
    if not isinstance(record,dict):
        return None
    prompt=record.get("p",record.get("prompt"))
    if not isinstance(prompt,str) or not prompt.strip():return None
    # All seeds of one prompt share a group; avoid treating generation seeds as independent.
    return digest(["generation_prompt"," ".join(prompt.split())])

def inventory(config):
    legacy=Path(config["paths"]["legacy_root"])
    meta_ai=_metadata(legacy/"data/raw/diffusiondb_part1") if (legacy/"data/raw/diffusiondb_part1").is_dir() else {}
    rows=[]
    for domain in ("real","ai"):
        index=legacy/"data/work"/f"index_{domain}.csv"
        if not index.is_file():raise Blocked("Missing ICISC work index: "+str(index))
        source_rows=list(csv.DictReader(index.open(encoding="utf-8-sig")))
        for r in progress(source_rows,"inventory legacy "+domain):
            p=legacy/"data/work"/domain/r["work_file"]
            prompt_key=_prompt_key(meta_ai,r["source_file"]) if domain=="ai" else None
            group=prompt_key or digest([domain,r["source_file"].casefold()])
            image=image_read(p)
            rows.append({"image_id":digest([domain,r["source_file"]])[:24],"group_id":group,
                         "domain":domain,"source_file":r["source_file"],"path":str(p.resolve()),
                         "historical":True,"sha256":file_hash(p),"phash":str(perceptual_hash(image)),
                         "height":image.shape[0],"width":image.shape[1],
                         "group_evidence":"prompt" if prompt_key else "source_id_only",
                         "upscaled_in_legacy":bool(int(r["resized"]))})
    legacy_ids={(r["domain"],r["source_file"].casefold()) for r in rows}
    for spec in config["data"]["fresh_sources"]:
        folder=Path(spec["path"])
        if not folder.is_dir():continue
        meta=_metadata(spec["metadata"])
        paths=sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".webp"))
        for p in progress(paths,"inventory fresh "+spec["domain"]):
            if (spec["domain"],p.name.casefold()) in legacy_ids:continue
            prompt_key=_prompt_key(meta,p.name) if spec["domain"]=="ai" else None
            if spec["domain"]=="ai" and prompt_key is None:
                raise Blocked("Fresh generated data need prompt metadata: "+p.name)
            image=image_read(p)
            group=prompt_key or digest([spec["domain"],p.name.casefold()])
            rows.append({"image_id":digest([spec["domain"],p.name])[:24],"group_id":group,
                         "domain":spec["domain"],"source_file":p.name,"path":str(p.resolve()),
                         "historical":False,"sha256":file_hash(p),"phash":str(perceptual_hash(image)),
                         "height":image.shape[0],"width":image.shape[1],
                         "group_evidence":"prompt" if prompt_key else "source_id_only",
                         "upscaled_in_legacy":False})
    if len({r["image_id"] for r in rows})!=len(rows):
        raise ValueError("Duplicate source ids in input roots")
    # Union content duplicates and shared prompts BEFORE split selection.
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[max(a,b)]=min(a,b)
    known={};pairs=[]
    for i,r in enumerate(rows):
        for key in ("group_id","sha256"):
            token=(key,r[key])
            if token in known:union(i,known[token])
            else:known[token]=i
    threshold=config["data"]["phash_distance"]
    for i in progress(range(len(rows)),"near-duplicate groups"):
        value=int(rows[i]["phash"])
        for j in range(i):
            if (value^int(rows[j]["phash"])).bit_count()<=threshold:
                union(i,j)
                pairs.append({"a":rows[i]["image_id"],"b":rows[j]["image_id"],"distance":(value^int(rows[j]["phash"])).bit_count()})
    members=defaultdict(list)
    for i,r in enumerate(rows):members[find(i)].append(r)
    result=[]
    for vals in members.values():
        historical=any(r["historical"] for r in vals)
        # One representative per group ensures image=independent group in evaluation.
        chosen=min(vals,key=lambda r:(not r["historical"],r["image_id"]))
        chosen=dict(chosen,group_id=digest(sorted(r["image_id"] for r in vals)),
                    historical=historical,group_members=[r["image_id"] for r in vals],
                    group_member_count=len(vals))
        result.append(chosen)
    return result,pairs,rows

def split_development(rows,config):
    out=[]
    fraction=config["data"]["validation_fraction"]
    for domain in ("real","ai"):
        selected=sorted([r for r in rows if r["domain"]==domain and r["historical"]],
                        key=lambda r:digest([config["training"]["seed"],r["group_id"]]))
        limit=config["data"]["development_limit_per_domain"]
        if limit is not None:selected=selected[:limit]
        pilot=config["data"]["pilot_per_domain"]
        if len(selected)<pilot+4:raise Blocked("Not enough independent development groups for "+domain)
        n_val=max(2,round((len(selected)-pilot)*fraction))
        for i,r in enumerate(selected):
            split="pilot" if i<pilot else ("validation" if i<pilot+n_val else "fit")
            out.append(dict(r,split=split))
    out.extend(dict(r,split="reserved_fresh") for r in rows if not r["historical"])
    return out

def assign_fresh(manifest,n_cal,n_test,seed_value):
    if n_cal<2 or n_test<2:raise ValueError("Fresh group counts must be >=2")
    out=[dict(r) for r in manifest if r["historical"]]
    for domain in ("real","ai"):
        fresh=sorted([r for r in manifest if not r["historical"] and r["domain"]==domain],
                     key=lambda r:digest([seed_value,r["group_id"]]))
        nc=n_cal//2+(domain=="real" and n_cal%2)
        nt=n_test//2+(domain=="real" and n_test%2)
        if len(fresh)<nc+nt:
            raise Blocked(f"{domain}: need {nc+nt} fresh groups, have {len(fresh)}")
        for i,r in enumerate(fresh):
            out.append(dict(r,split="calibration" if i<nc else ("test" if i<nc+nt else "reserve")))
    groups=[r["group_id"] for r in out]
    if len(set(groups))!=len(groups):raise ValueError("Cross-split group overlap")
    return out

