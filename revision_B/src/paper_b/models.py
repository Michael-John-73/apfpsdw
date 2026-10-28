"""Development-only fitting of count models, analytic gradients and frozen state."""
from __future__ import annotations
from collections import defaultdict
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp
from .common import digest, progress
from .probability import EPS, bb_pmf, binomial_pmf, poisson_binomial, log_bb, grad_log_bb, temperature_pmf

MODEL_NAMES=("global_binomial","source_binomial","perbit_product","beta_binomial",
             "feature_binomial","generic_bb","mechanistic_bb")
BASELINES=MODEL_NAMES[:-1]

def usable(rows):
    selected=[r for r in rows if r["role"]=="public_mark" and r["status"]=="ok"]
    if not selected:
        raise ValueError("No supported marked observations for count modeling")
    return selected

def weights_for(rows):
    counts=defaultdict(int)
    for r in rows: counts[r["group_id"]]+=1
    w=np.array([1/counts[r["group_id"]] for r in rows],float)
    return w/w.sum()

def columns_for(name,rows):
    if name in ("global_binomial","source_binomial","perbit_product","beta_binomial"):
        return []
    all_columns=sorted(rows[0]["features"])
    if any(sorted(r["features"])!=all_columns for r in rows):
        raise ValueError("Inconsistent pretransmission feature schema")
    if name=="generic_bb":
        return [c for c in all_columns if not c.startswith("mechanism_")]
    return all_columns

def design_matrix(rows,columns,mean=None,scale=None):
    if not columns:
        return np.ones((len(rows),1)),[],[]
    x=np.array([[r["features"][c] for c in columns] for r in rows],float)
    if not np.isfinite(x).all(): raise ValueError("Nonfinite model features")
    if mean is None:
        mean=x.mean(axis=0)
        scale=x.std(axis=0)
        scale=np.where(scale<1e-10,1.,scale)
    mean,scale=np.asarray(mean),np.asarray(scale)
    return np.column_stack([np.ones(len(x)),(x-mean)/scale]),mean.tolist(),scale.tolist()

def bb_objective(parameters,x,k,length,weights,l2):
    p=x.shape[1]
    eta=x@parameters[:p]
    rho=x@parameters[p:]
    sigmoid=expit(eta)
    mu=EPS+(1-2*EPS)*sigmoid
    concentration=np.logaddexp(0,rho)+2*EPS
    a,b=mu*concentration,(1-mu)*concentration
    logp=log_bb(k,length,a,b)
    da,db=grad_log_bb(k,length,a,b)
    mean_factor=concentration*(da-db)*(1-2*EPS)*sigmoid*(1-sigmoid)
    concentration_factor=(mu*da+(1-mu)*db)*expit(rho)
    penalty_mask=np.ones_like(parameters);penalty_mask[[0,p]]=0
    loss=-np.dot(weights,logp)+.5*l2*np.sum((parameters*penalty_mask)**2)
    gradient=np.r_[-x.T@(weights*mean_factor),-x.T@(weights*concentration_factor)]
    gradient+=l2*parameters*penalty_mask
    return float(loss),gradient

def binomial_objective(parameters,x,k,length,weights,l2):
    from scipy.special import gammaln
    eta=x@parameters
    p=expit(eta)
    combinatorial=gammaln(length+1)-gammaln(k+1)-gammaln(length-k+1)
    logp=combinatorial+k*eta-length*np.logaddexp(0,eta)
    mask=np.ones_like(parameters);mask[0]=0
    return (float(-weights@logp+.5*l2*np.sum((parameters*mask)**2)),
            x.T@(weights*(length*p-k))+l2*mask*parameters)

def _group_key(row):
    return row["variant"]+"|"+row["payload_id"]+"|"+row["condition_id"]

def fit_model(name,rows,l2,training,show_progress=True,compute=None):
    if name not in MODEL_NAMES: raise ValueError("Unknown model "+name)
    if any(r["split"] not in ("pilot","fit","validation") for r in rows):
        raise ValueError("Model fitting may use development rows only")
    data=usable(rows);w=weights_for(data)
    k=np.array([r["K"] for r in data],float)
    length=np.array([r["L"] for r in data],float)
    result={"name":name,"columns":columns_for(name,data),"l2":l2,"temperature":1.,
            "n_rows":len(data),"n_groups":len({r["group_id"] for r in data}),
            "train_input_hash":digest(data),"weighting":"equal group then equal supported marked row",
            "training":training}
    p_error=float(np.clip(np.dot(w,k)/np.dot(w,length),EPS,1-EPS))
    if name in ("global_binomial","source_binomial"):
        result.update(p=p_error,groups={},converged=True,parameter_count=1)
        if name=="source_binomial":
            groups=defaultdict(list)
            for i,r in enumerate(data):groups[_group_key(r)].append(i)
            for key,index in groups.items():
                ii=np.array(index);ww=w[ii]
                # Beta(1,1) smoothing is fixed, never estimated on validation/test.
                ww=ww/ww.sum()
                n=len({data[i]["group_id"] for i in index})
                result["groups"][key]=float((n*np.dot(ww,k[ii])+1)/(n*np.dot(ww,length[ii])+2))
            result["parameter_count"]+=len(result["groups"])
        return result
    if name=="perbit_product":
        groups=defaultdict(list);pooled=defaultdict(list)
        for r in data:
            groups[_group_key(r)].append(r)
            pooled[str(r["L"])].append(r)
        def probabilities(items):
            errors=np.array([r["errors"] for r in items],float)
            ww=weights_for(items)
            # Effective source count preserves a weak Beta(1,1) prior.
            n=len({r["group_id"] for r in items})
            return ((ww@errors*n+1)/(n+2)).tolist()
        group_probabilities={key:probabilities(val) for key,val in groups.items()}
        fallback={key:probabilities(val) for key,val in pooled.items()}
        result.update(groups=group_probabilities,fallback=fallback,converged=True,
                      parameter_count=sum(len(v) for v in group_probabilities.values())+sum(len(v) for v in fallback.values()))
        return result
    x,mean,scale=design_matrix(data,result["columns"])
    result.update(mean=mean,scale=scale)
    p=x.shape[1]
    is_bb=name!="feature_binomial"
    initial=np.zeros(p*(2 if is_bb else 1))
    initial[0]=np.log(p_error/(1-p_error))
    if is_bb:initial[p]=10.
    function=bb_objective if is_bb else binomial_objective
    batched=None
    if compute is not None:
        from .batch_backend import CountObjective
        batched=CountObjective(x,k,length,w,l2,is_bb,compute,function)
    def evaluate(parameters):
        return batched(parameters) if batched is not None else function(parameters,x,k,length,w,l2)
    history=[]
    bar=progress(range(training["maxiter"]),name) if show_progress else None
    last=np.array(initial)
    def callback(parameters):
        nonlocal last
        value,gradient=evaluate(parameters)
        history.append({"iteration":len(history)+1,"objective":value,
                        "gradient_norm":float(np.linalg.norm(gradient)),
                        "parameter_update":float(np.linalg.norm(parameters-last))})
        last=parameters.copy()
        if bar:
            bar.update(1)
            if len(history)%10==0:bar.set_postfix(loss=f"{value:.5f}")
    start=time.perf_counter()
    optimized=minimize(evaluate,initial,jac=True,method="L-BFGS-B",
                       bounds=[(-30.,30.)]*len(initial),callback=callback,
                       options={"maxiter":training["maxiter"],"ftol":training["ftol"],
                                "gtol":training["gtol"],"maxls":40})
    if bar:bar.close()
    value,gradient=evaluate(optimized.x)
    if not np.isfinite(optimized.x).all() or not np.isfinite(value):
        raise ValueError("Nonfinite optimization result")
    result.update(parameters=optimized.x.tolist(),initial_parameters=initial.tolist(),
                  converged=bool(optimized.success),optimizer_status=int(optimized.status),
                  optimizer_message=str(optimized.message),objective=value,
                  final_gradient_norm=float(np.linalg.norm(gradient)),
                  iterations=int(optimized.nit),elapsed_s=time.perf_counter()-start,
                  parameter_count=len(initial),history=history,
                  optimized_variables="mean regression; concentration regression for BB",
                  frozen_modules="watermark encoder/decoder and channel operators")
    if batched is not None:
        report=batched.report()
        if report["actual_device"]=="cuda":
            reference_value,reference_gradient=function(optimized.x,x,k,length,w,l2)
            np.testing.assert_allclose(value,reference_value,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(gradient,reference_gradient,rtol=1e-5,atol=1e-7)
            report.update(final_cpu_reference_check="PASS",objective_abs_error=abs(value-reference_value),
                          gradient_max_abs_error=float(np.max(np.abs(gradient-reference_gradient))))
        result.update(compute_config=compute,compute_report=report)
    return result

def predict_pmf(model,row,calibrated=True):
    n=int(row["L"]);name=model["name"]
    if name=="global_binomial":
        pmf=binomial_pmf(n,model["p"])
    elif name=="source_binomial":
        pmf=binomial_pmf(n,model["groups"].get(_group_key(row),model["p"]))
    elif name=="perbit_product":
        p=model["groups"].get(_group_key(row),model["fallback"].get(str(n)))
        if p is None:raise ValueError("No training-only per-bit fallback for this length")
        pmf=poisson_binomial(p)
    else:
        x,_,_=design_matrix([row],model["columns"],model["mean"],model["scale"])
        parameters=np.asarray(model["parameters"]);p=x.shape[1]
        eta=(x@parameters[:p]).item()
        mu=expit(eta) if name=="feature_binomial" else EPS+(1-2*EPS)*expit(eta)
        if name=="feature_binomial":
            pmf=binomial_pmf(n,mu)
        else:
            concentration=float(np.logaddexp(0,(x@parameters[p:]).item()))+2*EPS
            pmf=bb_pmf(n,mu*concentration,(1-mu)*concentration)
    return temperature_pmf(pmf,model.get("temperature",1.)) if calibrated else pmf

def predict_pmfs(model,rows,compute=None):
    from .batch_backend import predict_batches
    return predict_batches(model,rows,compute,predict_pmf)

def score_model(model,rows,compute=None):
    data=usable(rows)
    pmfs,report=predict_pmfs(model,data,compute)
    loss=np.array([-np.log(max(p[r["K"]],1e-300)) for p,r in zip(pmfs,data)])
    return float(weights_for(data)@loss)

def calibrate_model(model,rows,bounds,maxiter,compute=None):
    if any(r["split"]!="calibration" for r in rows):
        raise ValueError("Temperature may only be fitted on the calibration split")
    data=usable(rows);w=weights_for(data)
    pmfs,compute_report=predict_pmfs(dict(model,temperature=1.),data,compute)
    logps=[np.log(np.maximum(p,1e-300)) for p in pmfs]
    def objective(temp):
        return float(sum(ww*(-lp[r["K"]]/temp+logsumexp(lp/temp)) for ww,lp,r in zip(w,logps,data)))
    baseline=objective(1.)
    fit=minimize_scalar(objective,bounds=tuple(bounds),method="bounded",options={"maxiter":maxiter,"xatol":1e-6})
    if not fit.success:raise ValueError("Probability calibration did not converge")
    # The identity option is a predeclared candidate, preserving a sound fallback.
    chosen=float(fit.x) if fit.fun<baseline else 1.
    result=dict(model,temperature=chosen)
    report={"temperature":chosen,"identity_nll":baseline,"calibrated_nll":objective(chosen),
            "n_groups":len({r["group_id"] for r in data}),"bounds":bounds,
            "calibration_input_hash":digest(data),"compute":compute_report}
    return result,report

