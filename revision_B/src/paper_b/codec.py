"""Vectorized, instrumented raw-LL watermark operations and explicit channel paths.

The quantizer intentionally preserves the deployed uint8 cast and swapped-detail
path. Correct-detail and float32 paths are controlled interventions. No DCT is
used in this watermark carrier; DCT in JPEG/deduplication is a separate operation.
"""
from __future__ import annotations
import hashlib
import io
from time import perf_counter
import cv2
import numpy as np
import pywt
from PIL import Image

XL_BITS="101100111110110010010000011110111011000110011110"
V1_BITS="".join(f"{b:08b}" for b in b"StableDiffusionV1")
PAYLOADS={"xl":np.array(list(map(int,XL_BITS)),np.uint8),
          "v1":np.array(list(map(int,V1_BITS)),np.uint8)}

def _validate(image):
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError("Expected uint8 BGR image")
    if min(image.shape[:2])<8:
        raise ValueError("No complete 8x8 carrier")
    if image.shape[0]*image.shape[1]<65536:
        raise ValueError("Deployed encoder size guard: area must be >=65536")

def _blocks(ll):
    nr,nc=ll.shape[0]//4,ll.shape[1]//4
    return ll[:nr*4,:nc*4].reshape(nr,4,nc,4).transpose(0,2,1,3).reshape(nr,nc,16).copy()

def _channel(image,kind):
    if kind=="uint8":
        return cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    if kind!="float32":
        raise ValueError("dtype path must be uint8 or float32")
    # Reproduce SDXL tensor preprocessing, including float32 round-off.
    normalized=image.astype(np.float32)/np.float32(255)
    tensor=normalized*np.float32(2)-np.float32(1)
    incoming=np.float32(255)*(tensor/np.float32(2)+np.float32(.5))
    return cv2.cvtColor(incoming,cv2.COLOR_BGR2YUV)

def encode(image,bits,delta=36.,kind="uint8",swap=True):
    _validate(image)
    bits=np.asarray(bits,np.uint8).reshape(-1)
    if len(bits)<1 or not np.isin(bits,[0,1]).all() or not np.isfinite(delta) or delta<=0:
        raise ValueError("Binary nonempty payload and positive delta required")
    if kind=="float32" and image.shape[1]<256:
        raise ValueError("SDXL width skip; exclude this path from marked-count modeling")
    yuv=_channel(image,kind)
    h,w=image.shape[0]//4*4,image.shape[1]//4*4
    original_u=yuv[:h,:w,1].astype(np.float64).copy()
    ll,(horizontal,vertical,diagonal)=pywt.dwt2(yuv[:h,:w,1],"haar")
    blocks=_blocks(ll)
    nr,nc=blocks.shape[:2]
    if nr*nc<len(bits):raise ValueError("Payload exceeds complete carrier count")
    values=blocks[...,1:]
    mag=np.abs(values)
    pos=np.argmax(mag,axis=-1)+1
    old=np.take_along_axis(blocks,pos[...,None],axis=-1)[...,0]
    sorted_mag=np.partition(mag,-2,axis=-1)
    gap=np.abs(old)-sorted_mag[...,-2]
    b=bits[np.arange(nr*nc).reshape(nr,nc)%len(bits)]
    q=(np.abs(old)//delta+.25+.5*b)*delta
    e=q-np.abs(old)
    modified=np.where(old>=0,q,-q)
    generic_u=cv2.cvtColor(image,cv2.COLOR_BGR2YUV)[...,1].astype(float)
    ll_original=np.abs(ll.astype(float)).ravel()
    generic={
        "u_mean":float(generic_u.mean()),"u_std":float(generic_u.std()),
        "u_p10":float(np.quantile(generic_u,.1)),"u_p90":float(np.quantile(generic_u,.9)),
        "u_sat":float(np.mean((generic_u<=1)|(generic_u>=254))),
        "u_gradient":float(np.mean(np.abs(np.diff(generic_u,axis=0)))),
        "ll_mean":float(ll_original.mean()),"ll_std":float(ll_original.std()),
        "ll_p90":float(np.quantile(ll_original,.9))}
    np.put_along_axis(blocks,pos[...,None],modified[...,None],axis=-1)
    ll[:nr*4,:nc*4]=blocks.reshape(nr,nc,4,4).transpose(0,2,1,3).reshape(nr*4,nc*4)
    details=(vertical,horizontal,diagonal) if swap else (horizontal,vertical,diagonal)
    before_cast=pywt.idwt2((ll,details),"haar")
    precast_mse=float(np.mean((before_cast-original_u)**2))
    overflow=float(np.mean((before_cast<0)|(before_cast>255)))
    yuv[:h,:w,1]=before_cast
    postcast_mse=float(np.mean((yuv[:h,:w,1].astype(float)-original_u)**2))
    raw=cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
    raw_out=float(np.mean((raw<0)|(raw>255)))
    if kind=="float32":
        tensor_out=np.clip(2*(raw/np.float32(255)-np.float32(.5)),-1,1)
        marked=np.clip(np.rint((tensor_out+1)/2*255),0,255).astype(np.uint8)
    else:
        marked=raw
    counts=np.bincount(np.arange(nr*nc)%len(bits),minlength=len(bits))
    cutoff=np.floor(127*counts/255)+1
    mechanism={
        "gap_mean":float(gap.mean()),"gap_p10":float(np.quantile(gap,.1)),
        "gap_after_quant_p10":float(np.quantile(gap+e,.1)),
        "argmax_at_risk":float(np.mean(gap+e<=0)),
        "quant_error_mse":float(np.mean(e**2)),
        "precast_u_mse":precast_mse,"postcast_u_mse":postcast_mse,
        "u_overflow":overflow,"bgr_out_of_range":raw_out,
        "repeat_mean":float(counts.mean()),"repeat_std":float(counts.std()),
        "vote_cutoff_fraction":float(np.mean(cutoff/counts))}
    mse=float(np.mean((marked.astype(float)-image.astype(float))**2))
    trace={"generic":generic,"mechanism":mechanism,"n_blocks":nr*nc,
           "payload_one_fraction":float(bits.mean()),"mse_rgb":mse,
           "psnr":None if mse==0 else float(10*np.log10(255**2/mse)),
           "psnr_infinite":mse==0,"pre_block_pos":pos.ravel(),
           "embedded_block_bits":b.ravel()}
    return marked,trace

def decode(image,length,delta=36.):
    if type(length) is not int or length<1 or not np.isfinite(delta) or delta<=0:
        raise ValueError("Positive payload length and delta required")
    if image.shape[0]*image.shape[1]<65536:
        return {"status":"guard_rejected","bits":None,"count":None}
    _validate(image)
    yuv=cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    h,w=image.shape[0]//4*4,image.shape[1]//4*4
    ll,_=pywt.dwt2(yuv[:h,:w,1],"haar")
    blocks=_blocks(ll)
    pos=np.argmax(np.abs(blocks[...,1:]),axis=-1)+1
    coefficients=np.take_along_axis(blocks,pos[...,None],axis=-1)[...,0]
    hard=((np.abs(coefficients)%delta)>.5*delta).astype(np.uint8).ravel()
    idx=np.arange(len(hard))%length
    counts=np.bincount(idx,minlength=length)
    if np.any(counts==0):raise ValueError("Payload exceeds complete carrier count")
    means=np.bincount(idx,weights=hard,minlength=length)/counts
    return {"status":"ok","bits":(means*255>127).astype(np.uint8),
            "block_bits":hard,"block_pos":pos.ravel(),"vote_means":means}

def channel(image,stages,filename):
    out=image.copy()
    for stage in stages:
        operation=stage["op"]
        if operation=="jpeg":
            rgb=cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
            buffer=io.BytesIO()
            Image.fromarray(rgb).save(buffer,format="JPEG",quality=stage["quality"],
                                     subsampling=stage["subsampling"],optimize=False)
            buffer.seek(0)
            out=cv2.cvtColor(np.asarray(Image.open(buffer).convert("RGB")),cv2.COLOR_RGB2BGR)
        elif operation=="chroma":
            yuv=cv2.cvtColor(out,cv2.COLOR_BGR2YUV)
            factor_x,factor_y=stage["fx"],stage["fy"]
            for c in (1,2):
                h,w=yuv.shape[:2]
                reduced=cv2.resize(yuv[...,c],(max(1,round(w/factor_x)),max(1,round(h/factor_y))),interpolation=cv2.INTER_AREA)
                yuv[...,c]=cv2.resize(reduced,(w,h),interpolation=cv2.INTER_LINEAR)
            out=cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
        elif operation=="resize":
            h,w=out.shape[:2]
            out=cv2.resize(out,(max(1,round(w*stage["scale"])),max(1,round(h*stage["scale"]))),interpolation=cv2.INTER_AREA)
        elif operation=="crop":
            area=float(stage["area"])
            rng=np.random.default_rng(int(hashlib.sha256(f"{filename}|crop{area}".encode()).hexdigest()[:8],16))
            h,w=out.shape[:2]
            nh,nw=max(1,round(h*np.sqrt(area))),max(1,round(w*np.sqrt(area)))
            top=int(rng.integers(0,h-nh+1));left=int(rng.integers(0,w-nw+1))
            out=out[top:top+nh,left:left+nw].copy()
        elif operation=="blur":
            out=cv2.GaussianBlur(out,(5,5),stage["sigma"])
        elif operation=="noise":
            tag=f"{filename}|noise{stage['sigma']}"
            rng=np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:8],16))
            out=np.clip(out.astype(float)+rng.normal(0,255*stage["sigma"],out.shape),0,255).astype(np.uint8)
        else:
            raise ValueError("Unknown channel operation: "+operation)
    return out

def planned_geometry(height,width,stages):
    for s in stages:
        if s["op"]=="resize":
            height,width=max(1,round(height*s["scale"])),max(1,round(width*s["scale"]))
        elif s["op"]=="crop":
            height,width=max(1,round(height*np.sqrt(s["area"]))),max(1,round(width*np.sqrt(s["area"])))
    return height,width

def feature_dict(trace,variant,length,delta,stages):
    meta={"is_float":float(variant["kind"]=="float32"),"swap":float(variant["swap"]),
          "length":float(length),"delta":float(delta),
          "payload_one_fraction":trace["payload_one_fraction"],"path_length":float(len(stages)),
          "first_jpeg":float(bool(stages) and stages[0]["op"]=="jpeg"),
          "last_jpeg":float(bool(stages) and stages[-1]["op"]=="jpeg")}
    for operation in ("jpeg","chroma","resize","crop","blur","noise"):
        meta["n_"+operation]=float(sum(s["op"]==operation for s in stages))
    for key,default in [("quality",100),("subsampling",0),("scale",1.),("area",1.),("sigma",0.)]:
        matches=[s[key] for s in stages if key in s]
        meta["planned_"+key]=float(matches[-1]) if matches else default
    return dict(meta,**{"generic_"+k:v for k,v in trace["generic"].items()},
                **{"mechanism_"+k:v for k,v in trace["mechanism"].items()})

def encode_fast(image,bits,delta=36.,kind="uint8",swap=True):
    """Same transform without feature instrumentation, for a fair replay cost."""
    _validate(image)
    bits=np.asarray(bits,np.uint8)
    if bits.ndim!=1 or not len(bits) or not np.isin(bits,[0,1]).all() or delta<=0:
        raise ValueError("Invalid payload/delta")
    if kind=="float32" and image.shape[1]<256:raise ValueError("SDXL width skip")
    yuv=_channel(image,kind);h,w=image.shape[0]//4*4,image.shape[1]//4*4
    ll,(horizontal,vertical,diagonal)=pywt.dwt2(yuv[:h,:w,1],"haar")
    blocks=_blocks(ll);nr,nc=blocks.shape[:2]
    if nr*nc<len(bits):raise ValueError("Payload exceeds carriers")
    pos=np.argmax(np.abs(blocks[...,1:]),axis=-1)+1
    old=np.take_along_axis(blocks,pos[...,None],axis=-1)[...,0]
    q=(np.abs(old)//delta+.25+.5*bits[np.arange(nr*nc).reshape(nr,nc)%len(bits)])*delta
    np.put_along_axis(blocks,pos[...,None],np.where(old>=0,q,-q)[...,None],axis=-1)
    ll[:nr*4,:nc*4]=blocks.reshape(nr,nc,4,4).transpose(0,2,1,3).reshape(nr*4,nc*4)
    yuv[:h,:w,1]=pywt.idwt2((ll,(vertical,horizontal,diagonal) if swap else (horizontal,vertical,diagonal)),"haar")
    out=cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
    if kind=="float32":out=np.clip(np.rint((np.clip(2*(out/np.float32(255)-np.float32(.5)),-1,1)+1)/2*255),0,255).astype(np.uint8)
    return out

