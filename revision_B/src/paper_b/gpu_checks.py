"""CPU-verifiable batch math plus honest, optional real CUDA profiling."""
from __future__ import annotations
import time
import numpy as np
import torch
from .batch_backend import DEFAULTS,settings,MemoryPolicy,CountObjective,_gpu_pmf,retry_batches
from .models import bb_objective,binomial_objective,predict_pmf
from .common import Blocked,progress

def run_cpu_checks():
    rng=np.random.default_rng(409024);checks=[]
    x=np.column_stack([np.ones(513),rng.normal(0,.3,(513,4))]);k=rng.integers(0,49,513).astype(float)
    length=np.full(513,48.);weights=rng.uniform(.1,1.,513);weights/=weights.sum()
    for is_bb,function in ((True,bb_objective),(False,binomial_objective)):
        theta=rng.normal(0,.2,10 if is_bb else 5)
        if is_bb:theta[5]=10.
        expected=function(theta,x,k,length,weights,.07)
        for batch in (1,7,64,128,1024):
            c=dict(DEFAULTS,device="cpu",train_batch_rows=batch,min_batch_rows=1)
            objective=CountObjective(x,k,length,weights,.07,is_bb,c,function)
            loss,gradient=objective(theta)
            np.testing.assert_allclose(loss,expected[0],rtol=1e-11,atol=1e-11)
            np.testing.assert_allclose(gradient,expected[1],rtol=1e-9,atol=1e-10)
        checks.append({"name":("BB" if is_bb else "binomial")+" nondivisible microbatches preserve global weights and L2 once","status":"PASS"})
        objective=CountObjective(x,k,length,weights,0.,is_bb,dict(DEFAULTS,device="cpu"),function)
        # Execute the same Torch arithmetic on CPU. This is not a CUDA test.
        loss,gradient=objective._gpu(theta,73)
        reference=function(theta,x,k,length,weights,0.)
        np.testing.assert_allclose([loss],reference[:1],rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(gradient,reference[1],rtol=1e-7,atol=1e-8)
        checks.append({"name":("BB" if is_bb else "binomial")+" Torch kernel on CPU versus independent NumPy/SciPy objective","status":"PASS"})
    for n in (48,136):
        rows=[{"L":n,"features":{"x":float(v)},"variant":"u","payload_id":"xl" if n==48 else "v1","condition_id":"none"} for v in np.linspace(-1,1,9)]
        for name in ("global_binomial","source_binomial","beta_binomial","feature_binomial","generic_bb","mechanistic_bb"):
            columns=[] if name=="beta_binomial" else ["x"]
            model={"name":name,"p":.3,"groups":{},"temperature":1.7,"columns":columns,
                   "mean":[0.] if columns else [],"scale":[1.] if columns else [],
                   "parameters":([-.8,.2] if name=="feature_binomial" else ([-.8,.2,10.,.1] if columns else [-.8,10.]))}
            policy=MemoryPolicy(dict(DEFAULTS,device="cpu"),len(rows),2,n+1)
            actual=_gpu_pmf(model,rows,policy)
            np.testing.assert_allclose(actual,np.array([predict_pmf(model,r) for r in rows]),rtol=2e-9,atol=1e-11)
        for p in (0.,1.):
            model={"name":"global_binomial","p":p,"temperature":1.,"groups":{}}
            np.testing.assert_allclose(_gpu_pmf(model,rows,policy),np.array([predict_pmf(model,r) for r in rows]),atol=1e-12)
    checks.append({"name":"Torch PMFs on CPU: six families, both lengths, temperature and binomial endpoints","status":"PASS"})
    attempts=[];releases=[];array=np.arange(100)
    def partial(batch):
        attempts.append(batch);total=0
        for start in range(0,len(array),batch):
            if batch>8 and start>=batch:raise torch.cuda.OutOfMemoryError("injected software fixture")
            total+=int(array[start:start+batch].sum())
        return total
    actual,batch,events=retry_batches(partial,32,4,4,torch.cuda.OutOfMemoryError,lambda:releases.append(1))
    assert actual==4950 and batch==8 and attempts==[32,16,8] and len(releases)==2
    checks.append({"name":"injected OOM retries whole operation without dropped/double-counted rows","status":"PASS"})
    try:retry_batches(lambda b:(_ for _ in ()).throw(torch.cuda.OutOfMemoryError("injected")),4,4,2,torch.cuda.OutOfMemoryError,lambda:None)
    except Blocked:checks.append({"name":"minimum batch/exhausted OOM fails closed","status":"PASS"})
    else:raise AssertionError("OOM silently accepted")
    try:retry_batches(lambda b:(_ for _ in ()).throw(ValueError("real error")),8,4,2,torch.cuda.OutOfMemoryError,lambda:None)
    except ValueError:checks.append({"name":"non-OOM error is never masked as memory pressure","status":"PASS"})
    else:raise AssertionError("Real error swallowed")
    for changes in ({"dtype":"float16"},{"min_batch_rows":65536},{"unused":1},{"reserve_gib":0}):
        try:settings(dict(DEFAULTS,**changes))
        except ValueError:continue
        raise AssertionError("Invalid compute configuration accepted")
    checks.append({"name":"invalid/unused/unsafe compute settings rejected","status":"PASS"})
    if not torch.cuda.is_available():
        try:MemoryPolicy(dict(DEFAULTS,device="cuda"),4096)
        except Blocked:checks.append({"name":"explicit CUDA cannot silently fall back on a CPU-only interpreter","status":"PASS"})
        else:raise AssertionError("CUDA falsely claimed")
    print(f"[BATCH CHECKS] {len(checks)} groups PASS; CUDA exercised=False in these CPU checks",flush=True)
    return {"status":"PASS","checks":checks,"count":len(checks),"cuda_exercised":False,
            "scope":"CPU/software fixtures; injected OOM is not observed real VRAM exhaustion"}

def cuda_profile(compute):
    report={"cpu_batch_checks":run_cpu_checks(),"status":"NOT_RUN_CUDA","cuda_available":torch.cuda.is_available(),
            "scientific_data":False,"note":"Synthetic resource/equivalence probe, not hypothesis evidence"}
    if not torch.cuda.is_available():return report
    c=settings(compute);c["device"]="cuda"
    rng=np.random.default_rng(409024);n=65536;p=41
    x=np.column_stack([np.ones(n),rng.normal(0,.25,(n,p-1))]);length=np.where(np.arange(n)%2,48.,136.)
    k=np.floor(rng.uniform(0,1,n)*(length+1));w=np.full(n,1/n);theta=np.zeros(p*2);theta[0]=-.7;theta[p]=10.
    reference=bb_objective(theta,x,k,length,w,.01);candidates=[]
    cache_checks=[]
    for is_bb,function,parameters in ((True,bb_objective,theta),(False,binomial_objective,theta[:p])):
        objective=CountObjective(x[:513],k[:513],length[:513],w[:513],.01,is_bb,
                                 dict(c,train_batch_rows=128),function)
        expected=function(parameters,x[:513],k[:513],length[:513],w[:513],.01)
        cached=objective.pinned_inputs
        for _ in range(2):
            objective.last=None
            loss,gradient=objective(parameters)
            np.testing.assert_allclose(loss,expected[0],rtol=1e-7,atol=1e-7)
            np.testing.assert_allclose(gradient,expected[1],rtol=1e-6,atol=1e-7)
            assert objective.pinned_inputs is cached
        if c["pin_memory"]:
            assert cached is not None and all(t.is_pinned() and t.device.type=="cpu" for t in cached)
        cache_checks.append({"family":"BB" if is_bb else "binomial","rows":513,"batch":128,"status":"PASS"})
    report["pinned_input_cache_checks"]=cache_checks
    for batch in progress(sorted({min(c["train_batch_rows"],b) for b in (1024,4096,16384,32768,c["train_batch_rows"])}),"4090 actual CUDA batch probes"):
        local=dict(c,train_batch_rows=max(c["min_batch_rows"],batch))
        objective=CountObjective(x,k,length,w,.01,True,local,bb_objective)
        value,grad=objective(theta)
        np.testing.assert_allclose(value,reference[0],rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(grad,reference[1],rtol=1e-6,atol=1e-7)
        times=[]
        for _ in range(2):
            objective.last=None;torch.cuda.synchronize(0);start=time.perf_counter();objective(theta);torch.cuda.synchronize(0)
            times.append(time.perf_counter()-start)
        telemetry=objective.report()
        if telemetry["peak_reserved_gib"]>telemetry["budget_gib"]+.05:raise ValueError("Observed PyTorch reserved peak exceeds declared process budget")
        candidates.append({"batch_requested":batch,"median_full_objective_s":float(np.median(times)),
                           "rows_per_s":n/float(np.median(times)),"loss_abs_error":abs(value-reference[0]),
                           "gradient_max_abs_error":float(np.max(np.abs(grad-reference[1]))),**telemetry})
    # GPU PMF equivalence on both payload lengths, all numeric families.
    gpu_checks=[]
    for length_value in (48,136):
        rows=[{"L":length_value,"features":{"x":float(v)},"variant":"u","payload_id":"xl" if length_value==48 else "v1","condition_id":"none"} for v in np.linspace(-1,1,257)]
        for name in ("global_binomial","source_binomial","beta_binomial","feature_binomial","generic_bb","mechanistic_bb"):
            cols=[] if name=="beta_binomial" else ["x"]
            model={"name":name,"p":.3,"groups":{},"temperature":1.3,"columns":cols,"mean":[0.] if cols else [],"scale":[1.] if cols else [],
                   "parameters":([-.8,.2] if name=="feature_binomial" else ([-.8,.2,10.,.1] if cols else [-.8,10.]))}
            policy=MemoryPolicy(c,len(rows),2,length_value+1,c["predict_batch_rows"])
            result=policy.execute(lambda batch:np.concatenate([_gpu_pmf(model,rows[i:i+batch],policy) for i in range(0,len(rows),batch)]))
            expected=np.array([predict_pmf(model,r) for r in rows]);np.testing.assert_allclose(result,expected,rtol=1e-7,atol=1e-9)
            gpu_checks.append({"model":name,"length":length_value,"max_abs_error":float(np.max(np.abs(result-expected))),**policy.report()})
    best=min(candidates,key=lambda r:r["median_full_objective_s"])
    report.update(status="PASS_CUDA",candidates=candidates,pmf_checks=gpu_checks,
                  recommended_train_batch_rows=best["resolved_batch_rows"],
                  recommendation_policy="Within declared batch cap; apply to a NEW config/run before fitting, never mutate frozen settings",
                  scope="float64 count objective and PMFs; no image codec/generator throughput claim")
    return report
