"""Bounded float64 CUDA batches; exact full-objective accumulation, no AMP.

Only small feature/count tensors reach CUDA. Image codecs remain the verified
CPU reference. SciPy retains the same L-BFGS-B optimizer and coefficient bounds.
"""
from __future__ import annotations
import gc
import math
import time
import numpy as np
from .common import Blocked,progress

DEFAULTS={"device":"cpu","dtype":"float64","train_batch_rows":32768,
          "predict_batch_rows":4096,"min_batch_rows":128,"gpu_min_rows":2048,
          "vram_fraction":.75,"max_vram_gib":18.,"reserve_gib":4.,
          "pin_memory":True,"auto_batch":True,"oom_retries":8,
          "forecast_source_batch":32}
GIB=1024**3

def settings(value=None):
    if value is None:return dict(DEFAULTS)
    if set(value)!=set(DEFAULTS):
        raise ValueError("compute keys missing/unknown: "+str(set(value)^set(DEFAULTS)))
    result=dict(value)
    if result["device"] not in ("cpu","cuda","auto") or result["dtype"]!="float64":
        raise ValueError("compute.device must be cpu/cuda/auto and dtype float64")
    for key in ("train_batch_rows","predict_batch_rows","min_batch_rows","gpu_min_rows","forecast_source_batch"):
        if type(result[key]) is not int or result[key]<1:raise ValueError("Positive integer compute."+key+" required")
    if result["min_batch_rows"]>min(result["train_batch_rows"],result["predict_batch_rows"]):raise ValueError("Minimum batch exceeds configured batch")
    if type(result["oom_retries"]) is not int or not 0<=result["oom_retries"]<=16:raise ValueError("Invalid OOM retry budget")
    if any(type(result[k]) is not bool for k in ("pin_memory","auto_batch")):raise ValueError("Boolean compute controls required")
    if not np.isfinite([result[k] for k in ("vram_fraction","max_vram_gib","reserve_gib")]).all():raise ValueError("Finite VRAM budgets required")
    if not 0<result["vram_fraction"]<=.9 or result["max_vram_gib"]<=0 or result["reserve_gib"]<1:
        raise ValueError("VRAM fraction <=.9, positive cap and >=1GiB reserve required")
    return result

def retry_batches(function,batch,minimum,retries,oom_type,release,on_retry=None):
    """Retry the WHOLE operation: partial gradients/rows are never retained."""
    events=[]
    for attempt in range(retries+1):
        try:return function(batch),batch,events
        except oom_type as exc:
            exc.__traceback__=None
            if batch<=minimum or attempt>=retries:
                release()
                raise Blocked(f"CUDA OOM at batch={batch}; no rows skipped. Reduce resource demand in a new run") from None
            reduced=max(minimum,batch//2)
            events.append({"from":batch,"to":reduced,"attempt":attempt+1})
            release()
            if on_retry:on_retry(batch,reduced)
            batch=reduced
    raise AssertionError("Unreachable retry state")

class MemoryPolicy:
    def __init__(self,config,rows,features=1,output_width=1,requested=None):
        import torch
        self.torch=torch;self.c=settings(config);self.actual="cpu";self.reason="explicit CPU"
        self.budget=0;self.events=[];self.cuda_exercised=False;self.allocator_backend=None
        available=torch.cuda.is_available()
        if self.c["device"]=="cuda" and not available:raise Blocked("CUDA requested but this interpreter has no CUDA device")
        eligible=self.c["device"]=="cuda" or (self.c["device"]=="auto" and rows>=self.c["gpu_min_rows"])
        if self.c["device"]=="auto":self.reason="CUDA unavailable" if not available else "below gpu_min_rows; CPU avoids launch/transfer overhead"
        self.requested=requested or self.c["train_batch_rows"]
        self.batch=min(rows,self.requested)
        self.minimum=min(self.c["min_batch_rows"],self.batch)
        if available and eligible:
            self.allocator_backend=torch.cuda.memory.get_allocator_backend()
            if self.allocator_backend!="native":raise Blocked("Use PYTORCH_ALLOC_CONF=backend:native before process start so allocator caps and reserved-memory telemetry have the validated semantics")
            free,total=torch.cuda.mem_get_info(0)
            reserved=torch.cuda.memory_reserved(0)
            self.budget=int(min(total*self.c["vram_fraction"],self.c["max_vram_gib"]*GIB,
                                max(0,free-self.c["reserve_gib"]*GIB+reserved)))
            if self.budget<256*1024**2:raise Blocked("Less than 256MiB usable CUDA budget after the reserved headroom")
            torch.cuda.set_per_process_memory_fraction(self.budget/total,0)
            torch.cuda.reset_peak_memory_stats(0)
            self.actual="cuda";self.reason="explicit CUDA" if self.c["device"]=="cuda" else "large numeric batch and CUDA available"
            # Conservative scratch bound, not a claim of measured peak usage.
            per_row=8*(features+12*output_width+128)*4
            if self.c["auto_batch"]:
                current=torch.cuda.memory_allocated(0)
                fit=max(0,int((self.budget-current)*.5/per_row))
                if fit<self.minimum:raise Blocked("Estimated CUDA batch does not fit the declared budget")
                self.batch=min(self.batch,fit)
            self.start_free=free;self.total=total
            print(f"[CUDA BATCH] rows={rows} batch={self.batch} budget={self.budget/GIB:.2f}GiB reserve={self.c['reserve_gib']:.1f}GiB float64",flush=True)
        else:
            self.start_free=None;self.total=None

    def tensor(self,array):
        tensor=self.torch.from_numpy(np.ascontiguousarray(array,dtype=np.float64))
        if self.actual=="cuda":
            if self.c["pin_memory"]:tensor=tensor.pin_memory()
            tensor=tensor.to("cuda:0",non_blocking=self.c["pin_memory"])
        return tensor

    def release_oom(self):
        gc.collect()
        if self.actual=="cuda":self.torch.cuda.empty_cache()

    def execute(self,function):
        if self.actual=="cpu":return function(self.batch)
        value,self.batch,events=retry_batches(function,self.batch,self.minimum,self.c["oom_retries"],
                    self.torch.cuda.OutOfMemoryError,self.release_oom,
                    lambda a,b:print(f"[CUDA OOM] retry whole operation batch {a}->{b}; no dropped rows",flush=True))
        self.events.extend(events);self.cuda_exercised=True
        return value

    def report(self):
        r={"requested_device":self.c["device"],"actual_device":self.actual,"reason":self.reason,
           "dtype":"float64","requested_batch_rows":self.requested,"resolved_batch_rows":self.batch,
           "oom_reductions":self.events,"cuda_exercised":self.cuda_exercised,
           "budget_gib":self.budget/GIB,"reserve_gib":self.c["reserve_gib"],
           "images_on_cuda":False,"full_dataset_on_cuda":False,"allocator_backend":self.allocator_backend}
        if self.actual=="cuda":
            t=self.torch;t.cuda.synchronize(0);free,total=t.cuda.mem_get_info(0)
            r.update(device_name=t.cuda.get_device_name(0),total_gib=total/GIB,free_gib=free/GIB,
                     allocated_gib=t.cuda.memory_allocated(0)/GIB,reserved_gib=t.cuda.memory_reserved(0)/GIB,
                     peak_allocated_gib=t.cuda.max_memory_allocated(0)/GIB,peak_reserved_gib=t.cuda.max_memory_reserved(0)/GIB)
        return r

class CountObjective:
    def __init__(self,x,k,length,weights,l2,is_bb,compute,reference):
        self.x=x;self.k=k;self.length=length;self.weights=weights;self.l2=l2;self.is_bb=is_bb;self.reference=reference
        self.policy=MemoryPolicy(compute,len(x),x.shape[1]);self.calls=0;self.seconds=0.;self.last=None
        self.pinned_inputs=None
        arrays=(x,k,length,weights)
        pinned_bytes=sum(a.size*8 for a in arrays)
        if self.policy.actual=="cuda" and self.policy.c["pin_memory"] and pinned_bytes<=256*1024**2:
            self.pinned_inputs=tuple(
                self.policy.torch.from_numpy(np.array(a,dtype=np.float64,order="C",copy=True)).pin_memory()
                for a in arrays)

    def _cpu(self,parameters,batch):
        loss=0.;gradient=np.zeros_like(parameters)
        for start in range(0,len(self.x),batch):
            end=start+batch
            value,grad=self.reference(parameters,self.x[start:end],self.k[start:end],self.length[start:end],self.weights[start:end],0.)
            loss+=value;gradient+=grad
        return loss,gradient

    def _gpu(self,parameters,batch):
        t=self.policy.torch;p=self.x.shape[1]
        with t.inference_mode():
            theta=self.policy.tensor(parameters);total=t.zeros((),dtype=t.float64,device=self.policy.actual)
            gradient=t.zeros_like(theta)
            for start in range(0,len(self.x),batch):
                end=start+batch
                if self.pinned_inputs is None:
                    x=self.policy.tensor(self.x[start:end]);k=self.policy.tensor(self.k[start:end])
                    n=self.policy.tensor(self.length[start:end]);w=self.policy.tensor(self.weights[start:end])
                else:
                    x,k,n,w=(a[start:end].to("cuda:0",non_blocking=True) for a in self.pinned_inputs)
                eta=x@theta[:p];comb=t.lgamma(n+1)-t.lgamma(k+1)-t.lgamma(n-k+1)
                if self.is_bb:
                    sigmoid=t.sigmoid(eta);mu=1e-8+(1-2e-8)*sigmoid
                    rho=x@theta[p:];con=t.logaddexp(t.zeros_like(rho),rho)+2e-8
                    a,b=mu*con,(1-mu)*con
                    lp=comb+t.lgamma(a+k)+t.lgamma(b+n-k)-t.lgamma(a+b+n)-t.lgamma(a)-t.lgamma(b)+t.lgamma(a+b)
                    common=-t.digamma(a+b+n)+t.digamma(a+b)
                    da=t.digamma(a+k)-t.digamma(a)+common;db=t.digamma(b+n-k)-t.digamma(b)+common
                    mean=con*(da-db)*(1-2e-8)*sigmoid*(1-sigmoid)
                    concentration=(mu*da+(1-mu)*db)*t.sigmoid(rho)
                    gradient[:p].add_(-x.T@(w*mean));gradient[p:].add_(-x.T@(w*concentration))
                else:
                    lp=comb+k*eta-n*t.logaddexp(t.zeros_like(eta),eta)
                    gradient.add_(x.T@(w*(n*t.sigmoid(eta)-k)))
                total.add_(-(w*lp).sum())
                del x,k,n,w,eta,comb,lp
            return float(total.cpu()),gradient.cpu().numpy().copy()

    def __call__(self,parameters):
        parameters=np.asarray(parameters,dtype=np.float64)
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1],self.last[2].copy()
        start=time.perf_counter()
        value,gradient=self.policy.execute(lambda batch:(self._gpu if self.policy.actual=="cuda" else self._cpu)(parameters,batch))
        mask=np.ones_like(parameters);mask[0]=0
        if self.is_bb:mask[self.x.shape[1]]=0
        value+=.5*self.l2*np.sum((parameters*mask)**2);gradient+=self.l2*parameters*mask
        if not np.isfinite(value) or not np.isfinite(gradient).all():raise ValueError("Nonfinite batched objective/gradient")
        self.calls+=1;self.seconds+=time.perf_counter()-start
        self.last=(parameters.copy(),float(value),gradient.copy())
        return float(value),gradient

    def report(self):return dict(self.policy.report(),objective_evaluations=self.calls,objective_s=self.seconds,
                                 pinned_input_cache_bytes=0 if self.pinned_inputs is None else sum(a.numel()*a.element_size() for a in self.pinned_inputs),
                                 accumulation="full-data weighted SUM; L2 once; no per-batch renormalization; optimizer step after all rows")

def _gpu_pmf(model,rows,policy):
    t=policy.torch;n=int(rows[0]["L"]);name=model["name"]
    with t.inference_mode():
        k=t.arange(n+1,dtype=t.float64,device=policy.actual)[None,:]
        comb=t.lgamma(t.tensor(float(n+1),dtype=t.float64,device=policy.actual))-t.lgamma(k+1)-t.lgamma(n-k+1)
        if name in ("global_binomial","source_binomial"):
            ps=[model.get("groups",{}).get(r["variant"]+"|"+r["payload_id"]+"|"+r["condition_id"],model["p"]) for r in rows]
            prob=policy.tensor(np.array(ps))[:,None]
            lp=comb+t.xlogy(k,prob)+t.xlogy(n-k,1-prob)
        else:
            columns=model["columns"]
            x=np.array([[r["features"][c] for c in columns] for r in rows],float)
            if columns:x=(x-np.asarray(model["mean"]))/np.asarray(model["scale"])
            x=np.column_stack([np.ones(len(rows)),x]);xx=policy.tensor(x);theta=policy.tensor(np.array(model["parameters"]))
            p=x.shape[1];mu=t.sigmoid(xx@theta[:p])[:,None]
            if name=="feature_binomial":lp=comb+t.xlogy(k,mu)+t.xlogy(n-k,1-mu)
            else:
                mu=1e-8+(1-2e-8)*mu;rho=xx@theta[p:];con=t.logaddexp(t.zeros_like(rho),rho)[:,None]+2e-8
                a,b=mu*con,(1-mu)*con
                lp=comb+t.lgamma(a+k)+t.lgamma(b+n-k)-t.lgamma(a+b+n)-t.lgamma(a)-t.lgamma(b)+t.lgamma(a+b)
        lp=lp-t.logsumexp(lp,dim=1,keepdim=True)
        lp=t.clamp(lp,min=math.log(1e-300))/model.get("temperature",1.)
        values=t.softmax(lp,dim=1)
        if not bool(t.isfinite(values).all()):raise ValueError("Nonfinite CUDA count PMF")
        return values.cpu().numpy().copy()

def predict_batches(model,rows,compute,reference):
    if not rows:return [],{"actual_device":"cpu","reason":"empty input","cuda_exercised":False}
    c=settings(compute);name=model["name"]
    if compute is None or name=="perbit_product":
        result=[reference(model,r) for r in rows]
        return result,{"actual_device":"cpu","reason":"legacy scalar path" if compute is None else "per-bit convolution kept as CPU reference",
                       "rows":len(rows),"cuda_exercised":False}
    policy=MemoryPolicy(c,len(rows),len(model.get("columns",[]))+1,max(r["L"]+1 for r in rows),c["predict_batch_rows"])
    start=time.perf_counter()
    def run(batch):
        result=[None]*len(rows)
        # Group only by payload length; never pad48 to136 or change original order.
        for length in sorted({r["L"] for r in rows}):
            indices=[i for i,r in enumerate(rows) if r["L"]==length]
            for off in range(0,len(indices),batch):
                ii=indices[off:off+batch];part=[rows[i] for i in ii]
                values=_gpu_pmf(model,part,policy) if policy.actual=="cuda" else [reference(model,r) for r in part]
                for i,value in zip(ii,values):result[i]=np.array(value,copy=True)
        return result
    result=policy.execute(run)
    return result,dict(policy.report(),rows=len(rows),elapsed_s=time.perf_counter()-start,
                       grouping="payload length; original trial order restored",autograd=False)
