"""Explicit bounded public-data downloads under the configured /workspace root."""
from pathlib import Path
from urllib.request import urlopen,Request
from zipfile import ZipFile
import shutil
from .common import Blocked,progress,file_hash,write_json,now

def acquire(s,parts):
    if s.c["environment"]["profile"]!="runpod":raise Blocked("Bulk acquisition is intended for /workspace on RunPod; local existing-data diagnostics do not download datasets")
    root=Path(s.c["paths"]["raw_root"]);root.mkdir(parents=True,exist_ok=True)
    if not str(root.resolve()).startswith("/workspace/"):raise ValueError("Storage outside /workspace")
    requests=[]
    for part in parts:
        if not 2<=part<=9999:raise ValueError("Fresh DiffusionDB parts start at 2, historical part1 is excluded")
        name=f"part-{part:06d}.zip"
        requests.append((f"https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/{name}",root/name,root/f"diffusiondb_part{part}"))
    records=[]
    for url,archive,target in requests:
        if not archive.is_file():
            partial=archive.with_suffix(".zip.partial")
            # A partial transfer is restarted, never treated as a completed archive.
            with urlopen(Request(url,headers={"User-Agent":"PaperB reproducibility/1"}),timeout=60) as response:
                size=int(response.headers.get("Content-Length",0))
                if size and shutil.disk_usage(root).free<3*size:raise Blocked("Insufficient storage for archive+extraction")
                with partial.open("wb") as out,progress([],"download "+archive.name,total=size or None) as bar:
                    while True:
                        chunk=response.read(1<<20)
                        if not chunk:break
                        out.write(chunk);bar.update(len(chunk))
            partial.replace(archive)
        target.mkdir(exist_ok=True)
        with ZipFile(archive) as z:
            members=z.infolist()
            if sum(x.file_size for x in members)>20*(1<<30):raise ValueError("Unexpected archive expansion")
            for item in progress(members,"extract "+archive.name):
                destination=(target/item.filename).resolve()
                if not destination.is_relative_to(target.resolve()):raise ValueError("Unsafe archive member")
                if item.is_dir():destination.mkdir(parents=True,exist_ok=True);continue
                data=z.read(item)
                if destination.exists():
                    if destination.read_bytes()!=data:raise ValueError("Existing extracted file differs; preserve and investigate")
                else:
                    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
        records.append({"url":url,"archive":str(archive),"archive_sha256":file_hash(archive),"folder":str(target),
                        "time":now(),"publisher_checksum_verified":False,"note":"local SHA256 provides provenance; not an authenticated publisher checksum"})
    write_json(s.root/"acquisition.json",records)
    s.finish("acquire",["acquisition.json"])
