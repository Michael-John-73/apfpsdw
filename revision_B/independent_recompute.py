"""Independent raw-to-metric recomputation using SciPy, without pipeline models."""
from pathlib import Path
from collections import defaultdict
import argparse,json,math,sys
import numpy as np
from scipy.stats import binom,betabinom,beta
from scipy.special import expit,logsumexp
from tqdm import tqdm
if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8",errors="replace")
p=argparse.ArgumentParser();p.add_argument("--run",required=True);a=p.parse_args();root=Path(a.run)
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
def reference(payload):
    return np.array([int(b) for b in ("101100111110110010010000011110111011000110011110" if payload=="xl" else "".join(format(c,"08b") for c in b"StableDiffusionV1"))])
def pmf(model,row):
    name=model["name"];n=row["L"];k=np.arange(n+1)
    key="|".join(row[x] for x in ("variant","payload_id","condition_id"))
    if name in ("global_binomial","source_binomial"):
        values=binom.pmf(k,n,model.get("groups",{}).get(key,model["p"]))
    elif name=="perbit_product":
        ps=model["groups"].get(key,model["fallback"][str(n)]);values=np.ones(1)
        for prob in ps:values=np.convolve(values,[1-prob,prob])
    else:
        x=np.array([row["features"][c] for c in model["columns"]])
        x=np.r_[1.,(x-np.asarray(model["mean"]))/np.asarray(model["scale"])]
        theta=np.array(model["parameters"]);size=len(x);mu=expit(x@theta[:size])
        if name=="feature_binomial":values=binom.pmf(k,n,mu)
        else:
            mu=1e-8+(1-2e-8)*mu;concentration=np.logaddexp(0,x@theta[size:])+2e-8
            values=betabinom.pmf(k,n,mu*concentration,(1-mu)*concentration)
    values=values/values.sum();logp=np.log(np.maximum(values,1e-300))/model.get("temperature",1.)
    return np.exp(logp-logsumexp(logp))
checks=[];splits={};forecast_pmf_error=0.;forecast_nll_error=0.
for split in ("pilot","fit","validation","calibration","test"):
    path=root/(split+"_rows.jsonl")
    if not path.exists():continue
    data=rows(path);splits[split]=data
    for r in tqdm(data,desc="independent bits "+split,file=sys.stdout):
        if r["bits"] is None:
            assert r["K"] is None and r["exact"]==r["tau_accept"]==0
        else:
            errors=np.asarray(r["bits"])!=reference(r["payload_id"])
            assert int(errors.sum())==r["K"] and errors.astype(int).tolist()==r["errors"]
            assert r["exact"]==int(not errors.any()) and r["tau_accept"]==int(errors.sum()<=math.floor(len(errors)/8))
    checks.append("raw bit/count/event "+split)
if (root/"selection.json").exists():
    table=read(root/"selection.json")["comparison"]
    for entry in tqdm(table,desc="independent SciPy validation PMFs",file=sys.stdout):
        model=read(root/"models"/(entry["id"]+".json"));groups=defaultdict(list)
        for r in splits["validation"]:
            if r["role"]=="public_mark" and r["K"] is not None:
                groups[r["group_id"]].append(-math.log(max(pmf(model,r)[r["K"]],1e-300)))
        value=np.mean([np.mean(v) for v in groups.values()])
        np.testing.assert_allclose(value,entry["validation_nll"],atol=1e-7,rtol=1e-7)
    checks.append("all validation candidates versus independent SciPy PMFs")
    selected=read(root/"selected_models.json")
    for model in selected.values():
        if not model.get("columns"):continue
        x=np.array([[r["features"][c] for c in model["columns"]] for r in splits["fit"] if r["role"]=="public_mark" and r["K"] is not None])
        scale=x.std(axis=0);scale[scale<1e-10]=1
        np.testing.assert_allclose(model["mean"],x.mean(axis=0),atol=1e-12)
        np.testing.assert_allclose(model["scale"],scale,atol=1e-12)
    checks.append("normalization learned from fit-only features")
if (root/"calibrated_models.json").exists():
    before=read(root/"selected_models.json");after=read(root/"calibrated_models.json")
    for key in before:
        b=dict(before[key]);c=dict(after[key]);b.pop("temperature");c.pop("temperature");assert b==c
    checks.append("calibration changes temperature only")
if (root/"evaluation.json").exists():
    ev=read(root/"evaluation.json");fc={r["trial_id"]:r for r in rows(root/"forecasts.jsonl")}
    cfg=read(root/"resolved_config.json");stats=cfg["statistics"];allrows=splits["test"]
    frozen_models=read(root/"calibrated_models.json")
    for row in tqdm([r for r in allrows if r["role"]=="public_mark"],desc="independent frozen-model forecast PMFs",file=sys.stdout):
        for name,model in frozen_models.items():
            independent=pmf(model,row);observed=np.array(fc[row["trial_id"]]["models"][name]["pmf"])
            np.testing.assert_allclose(observed,independent,rtol=1e-6,atol=1e-12)
            forecast_pmf_error=max(forecast_pmf_error,float(np.max(np.abs(observed-independent))))
            if row["K"] is not None:
                k=row["K"];error=abs(math.log(max(observed[k],1e-300))-math.log(max(independent[k],1e-300)))
                if error>1e-5:raise AssertionError("Frozen forecast realized-count NLL differs from independent CPU reference")
                forecast_nll_error=max(forecast_nll_error,error)
    checks.append("every frozen forecast PMF/realized NLL recomputed from saved coefficients and raw prechannel features")
    for panel,expected in ev["panels"].items():
        groups=defaultdict(list)
        for r in allrows:
            if r["panel"]!=panel or r["role"]!="public_mark" or r["K"] is None:continue
            f=fc[r["trial_id"]]["models"];k=r["K"]
            groups[r["group_id"]].append(math.log(max(f["proposed"]["pmf"][k],1e-300))-math.log(max(f["baseline"]["pmf"][k],1e-300)))
        values=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(stats["seed"]);draws=[]
        for start in range(0,stats["bootstrap_reps"],50):
            n=min(50,stats["bootstrap_reps"]-start)
            draws.extend(values[rng.integers(0,len(values),size=(n,len(values)))].mean(axis=1))
        tail=(1-stats["confidence"])/2;lo,hi=np.quantile(draws,[tail,1-tail]);ci=expected["paired_nll_gain"]
        np.testing.assert_allclose([values.mean(),lo,hi],[ci["estimate"],ci["low"],ci["high"]],atol=1e-10)
        selected=[r for r in allrows if r["role"]=="public_mark" and r["panel"]==panel]
        cells=defaultdict(list)
        for r in selected:
            f=fc[r["trial_id"]]["models"]["proposed"]
            cells[(r["variant"],r["payload_id"],r["condition_id"])].append([f["p_exact"]-r["exact"],f["p_tau"]-r["tau_accept"]])
        error=np.mean([np.abs(np.mean(v,axis=0)).mean() for v in cells.values()])
        np.testing.assert_allclose(error,expected["acceptance_probability_mae"]["estimate"],atol=1e-12)
        groupids=sorted({r["group_id"] for r in selected});keys=sorted(cells)
        gi={g:i for i,g in enumerate(groupids)};ki={k:i for i,k in enumerate(keys)}
        matrix=np.full((len(groupids),len(keys),2),np.nan)
        for r in selected:
            f=fc[r["trial_id"]]["models"]["proposed"]
            matrix[gi[r["group_id"]],ki[(r["variant"],r["payload_id"],r["condition_id"])]]=[f["p_exact"]-r["exact"],f["p_tau"]-r["tau_accept"]]
        assert np.isfinite(matrix).all();rng=np.random.default_rng(stats["seed"]);errors=[]
        for _ in range(stats["bootstrap_reps"]):
            index=rng.integers(0,len(groupids),len(groupids))
            errors.append(np.abs(matrix[index].mean(axis=0)).mean())
        ml,mh=np.quantile(errors,[tail,1-tail]);mae=expected["acceptance_probability_mae"]
        np.testing.assert_allclose([ml,mh],[mae["low"],mae["high"]],atol=1e-12)
        d=cfg["design"]
        verdict="SUPPORTED" if lo>d["delta_nll"] and (panel=="primary" or mh<=d["acceptance_mae"]) else ("NOT_SUPPORTED" if hi<=d["delta_nll"] or (panel!="primary" and ml>d["acceptance_mae"]) else "INCONCLUSIVE")
        assert verdict==expected["verdict"]
    lookup={(r["variant"],r["payload_id"],r["condition_id"],r["role"]):[] for r in allrows}
    for r in allrows:lookup[(r["variant"],r["payload_id"],r["condition_id"],r["role"])].append(r)
    for rate in ev["operating_rates"]:
        items=lookup[(rate["variant"],rate["payload"],rate["condition"],rate["role"])]
        assert rate["exact"]["n"]==len(items) and rate["exact"]["k"]==sum(r["exact"] for r in items)
        assert rate["tau"]["k"]==sum(r["tau_accept"] for r in items)
        for rule in ("exact","tau"):
            item=rate[rule];n=item["n"];k=item["k"];t=(1-item["confidence"])/2
            lo=0 if not k else beta.ppf(t,k,n-k+1);hi=1 if k==n else beta.ppf(1-t,k+1,n-k)
            np.testing.assert_allclose([lo,hi],[item["low"],item["high"]],atol=1e-12)
    checks.append("frozen forecasts plus raw labels -> panel NLL/paired CI/MAE CI/verdicts/operating denominators and intervals")
if (root/"cost.json").exists():
    cost=read(root/"cost.json");timings=rows(root/"cost_rows.jsonl")
    for key in ("predict","replay"):
        values=[r[key] for r in timings];summary=cost["summary"][key]
        np.testing.assert_allclose([np.median(values),np.quantile(values,.95),np.sum(values)],
                                   [summary["median_s"],summary["p95_s"],summary["sum_s"]],atol=1e-12)
    checks.append("full cost median/p95/sum from independent raw durations")
result={"status":"PASS","checks":checks,"rows":sum(len(v) for v in splits.values()),
        "forecast_max_abs_pmf_error":forecast_pmf_error,"forecast_max_realized_nll_error":forecast_nll_error,
        "independent_engine":"SciPy binom/betabinom and separate convolution; no pipeline probability/model/evaluate imports",
        "confirmation_present":"test" in splits,"limitations":"Source/library equivalence and mathematical gradients use the separate recorded checks; scientific independence depends on original source metadata."}
(root/"independent_recompute.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
