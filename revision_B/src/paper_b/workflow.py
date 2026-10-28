"""Learning, selection, calibration and genuinely separate confirmation gates."""
from collections import defaultdict,Counter
from pathlib import Path
import copy
import math
import time
import numpy as np
from scipy.stats import norm
from . import codec
from .checks import run_checks
from .common import Blocked,digest,read_json,write_json,read_rows,save_rows,progress,now,file_hash,append_json
from .config import confirmatory_requirements
from .data import assign_fresh,image_read
from .models import MODEL_NAMES,BASELINES,fit_model,score_model,predict_pmf,predict_pmfs,calibrate_model,usable
from .probability import acceptance,paired_interval,exact_ci

def mathcheck(s):
    report=run_checks();write_json(s.root/"mathcheck.json",report)
    s.finish("mathcheck",["mathcheck.json"])

def _candidates(s,fit,validation,folder):
    table=[];models={}
    jobs=[(name,l2) for name in MODEL_NAMES for l2 in (s.c["training"]["l2_candidates"] if name in MODEL_NAMES[3:] else [0.])]
    for name,l2 in progress(jobs,"candidate models"):
        model=fit_model(name,fit,l2,s.c["training"],compute=s.c.get("compute"))
        value=score_model(model,validation,s.c.get("compute")) if model["converged"] else None
        label=name+"_"+str(l2)
        write_json(s.root/folder/(label+".json"),model)
        models[label]=model
        table.append({"id":label,"name":name,"l2":l2,"converged":model["converged"],
                      "validation_nll":value,"parameters":model["parameter_count"],
                      "fit_s":model.get("elapsed_s",0.)})
    # Nonconvergence is an implementation/optimization issue, never a winning poor baseline.
    failures=[r["id"] for r in table if not r["converged"]]
    write_json(s.root/folder/"comparison.json",table)
    if failures:raise Blocked("Required candidates did not converge; revise optimizer on development only: "+", ".join(failures))
    def rank(r):return (round(r["validation_nll"],10),r["parameters"],-r["l2"],r["id"])
    best=min((r for r in table if r["name"] in BASELINES),key=rank)
    proposed=min((r for r in table if r["name"]=="mechanistic_bb"),key=rank)
    return table,{"baseline":models[best["id"]],"proposed":models[proposed["id"]]}, {"baseline":best,"proposed":proposed}

def _group_differences(models,rows,compute=None):
    groups=defaultdict(list)
    data=usable(rows)
    a_pmfs,_=predict_pmfs(models["baseline"],data,compute)
    b_pmfs,_=predict_pmfs(models["proposed"],data,compute)
    for r,ap,bp in zip(data,a_pmfs,b_pmfs):
        a=ap[r["K"]];b=bp[r["K"]]
        groups[r["group_id"]].append(float(np.log(max(b,1e-300))-np.log(max(a,1e-300))))
    return {g:float(np.mean(v)) for g,v in groups.items()}

def pilot(s):
    s.require("measure-pilot")
    if s.state["test_accessed"]:raise Blocked("No pilot revision after test access")
    rows=read_rows(s.root/"pilot_rows.jsonl")
    groups=sorted({r["group_id"] for r in rows})
    if len(groups)<8:raise Blocked("Pilot requires at least eight independent groups")
    hold=set(groups[::2]);fit=[r for r in rows if r["group_id"] not in hold];val=[r for r in rows if r["group_id"] in hold]
    table,models,selected=_candidates(s,fit,val,"pilot_models")
    differences=_group_differences(models,val,s.c.get("compute"))
    report={"scope":"pilot planning only; held-out pilot selected model is optimistic and not confirmatory evidence",
            "fit_groups":len(groups)-len(hold),"planning_groups":len(hold),"selection":selected,
            "group_difference_sd":float(np.std(list(differences.values()),ddof=1)),
            "mean_difference_descriptive":float(np.mean(list(differences.values()))),
            "supported_marked_rows":len(usable(rows)),"guard_refusals":sum(r["status"]=="guard_rejected" for r in rows),
            "median_encode_feature_s":float(np.median([r["encode_and_features_s"] for r in rows])),
            "median_channel_decode_s":float(np.median([r["channel_and_decode_s"] for r in rows])),
            "criterion":"planning uses SD and prespecified effect separation; observed gain is never the required effect"}
    write_json(s.root/"pilot_report.json",report)
    s.finish("pilot",["pilot_report.json"])

def plan(s):
    s.require("pilot","prepare")
    if "plan" in s.state["stages"] and s.state["stages"]["plan"]["status"]=="TECH_PASS":
        raise ValueError("Plan is already committed; do not reshuffle fresh groups")
    if s.state["test_accessed"]:raise Blocked("Cannot plan after test access")
    gaps=confirmatory_requirements(s.c)
    # test_groups may be computed here, all practical margins must already be justified.
    gaps=[x for x in gaps if x!="design.test_groups"]
    if gaps:raise Blocked("Unresolved confirmatory criteria: "+", ".join(gaps))
    d=s.c["design"];p=read_json(s.root/"pilot_report.json")
    z=norm.ppf((1+s.c["statistics"]["confidence"])/2);zp=norm.ppf(d["power"])
    sd=p["group_difference_sd"]
    # Factor 2 is a declared variance sensitivity, not a claim of guaranteed transfer power.
    n_effect=math.ceil(((z+zp)*2*sd/d["delta_nll"])**2)
    n_precision=math.ceil(z*z*.25/d["proportion_half_width"]**2)
    required=max(40,n_effect,n_precision)
    proposed=d["test_groups"] or required
    report={"minimum_effect_nll":d["delta_nll"],"effect_rationale":d["delta_justification"],
            "true_effect_for_power":2*d["delta_nll"],"separation_from_null":d["delta_nll"],
            "pilot_sd":sd,"sd_sensitivity_factor":2,"power":d["power"],"confidence":s.c["statistics"]["confidence"],
            "n_effect":n_effect,"n_worst_case_proportion_precision":n_precision,"minimum_test_groups":required,
            "planned_test_groups":proposed,"calibration_groups":d["calibration_groups"],
            "budget_groups":d["max_test_groups"],"status":"HOLD",
            "limit":"Pilot model selection is optimistic; doubling SD is a sensitivity. Unseen-path variance/power is not proven. Final CI may be inconclusive.",
            "acceptance_mae":d["acceptance_mae"],"mae_rationale":d["mae_justification"]}
    write_json(s.root/"sample_plan.json",report)
    if proposed<required or proposed>d["max_test_groups"]:
        raise Blocked("Required precision/power exceeds declared sample allocation/budget; see sample_plan.json. Do not relax margins to force PASS")
    audit=read_json(s.root/"data_audit.json")
    if audit["legacy_ai_group_evidence"].get("source_id_only",0):
        raise Blocked("Historical generated-image prompt metadata missing; fresh prompt independence cannot be verified")
    manifest=assign_fresh(read_json(s.root/"data_manifest.json"),d["calibration_groups"],proposed,s.c["statistics"]["seed"])
    s.c["design"]["test_groups"]=proposed
    write_json(s.root/"data_manifest.json",manifest)
    write_json(s.root/"resolved_config.json",s.c)
    report["status"]="READY_FOR_CALIBRATION_NOT_TEST";write_json(s.root/"sample_plan.json",report)
    s.finish("plan",["data_manifest.json","resolved_config.json","sample_plan.json"])

def select(s):
    s.require("measure-fit","measure-validation","pilot")
    if s.state["test_accessed"]:raise Blocked("Model reselection after test access prohibited")
    if s.state["stages"].get("freeze",{}).get("status")=="TECH_PASS":raise Blocked("Frozen model cannot be reselected")
    fit=read_rows(s.root/"fit_rows.jsonl");val=read_rows(s.root/"validation_rows.jsonl")
    table,models,selection=_candidates(s,fit,val,"models")
    report={"comparison":table,"selection":selection,"rule":"validation group-balanced count NLL; 1e-10 tie -> fewer parameters -> stronger L2 -> id",
            "learning_data":"fit only; pilot and validation never refitted into final weights",
            "baseline_is_strongest_declared":True,"technical_readiness":"DEV_READY",
            "hypothesis_support":"NOT_EVALUATED","heldout_gain_descriptive":selection["baseline"]["validation_nll"]-selection["proposed"]["validation_nll"]}
    write_json(s.root/"selected_models.json",models);write_json(s.root/"selection.json",report)
    s.finish("select",["selected_models.json","selection.json"])

def calibrate(s):
    s.require("select","plan","measure-calibration")
    if s.state["test_accessed"] or s.state["stages"].get("freeze",{}).get("status")=="TECH_PASS":raise Blocked("No calibration after freeze/test")
    models=read_json(s.root/"selected_models.json");rows=read_rows(s.root/"calibration_rows.jsonl")
    calibrated={};reports={}
    for name,model in progress(models.items(),"fixed-model probability calibration"):
        calibrated[name],reports[name]=calibrate_model(model,rows,s.c["calibration"]["temperature_bounds"],s.c["calibration"]["maxiter"],s.c.get("compute"))
    write_json(s.root/"calibrated_models.json",calibrated);write_json(s.root/"calibration.json",reports)
    s.finish("calibrate",["calibrated_models.json","calibration.json"])

def freeze(s):
    s.require("environment","source-audit","mathcheck","prepare","pilot","plan","select","calibrate")
    if s.c.get("compute",{}).get("device","cpu")!="cpu" and s.c["environment"]["require_cuda"]:
        s.require("gpu-profile")
        if read_json(s.root/"gpu_profile.json")["status"]!="PASS_CUDA":raise Blocked("Actual CUDA equivalence/memory profile must pass before freeze")
    if confirmatory_requirements(s.c):raise Blocked("Unresolved design criteria")
    if s.state["test_accessed"]:raise Blocked("Cannot create a new freeze after opening test")
    manifest=read_json(s.root/"data_manifest.json")
    if any(r["historical"] for r in manifest if r["split"] in ("test","calibration")):
        raise ValueError("Historical image in independent split")
    test=[r for r in manifest if r["split"]=="test"]
    if len(test)!=s.c["design"]["test_groups"]:raise ValueError("Locked test allocation mismatch")
    panels={c["panel"] for c in s.c["measurement"]["conditions"]}
    if panels!={"primary","transfer_order","transfer_length"}:raise Blocked("H_BT requires both independent order and length panels")
    primary=[c["stages"] for c in s.c["measurement"]["conditions"] if c["panel"]=="primary"]
    for c in s.c["measurement"]["conditions"]:
        if c["panel"]=="primary":continue
        if any(digest(c["stages"])==digest(p) for p in primary):raise ValueError("Transfer path duplicates a training path")
        if c["panel"]=="transfer_order" and not any(sorted(map(digest,c["stages"]))==sorted(map(digest,p)) and c["stages"]!=p for p in primary):
            raise ValueError("Order panel lacks the same operation multiset in primary")
        if c["panel"]=="transfer_length" and len(c["stages"])<=max(map(len,primary)):
            raise ValueError("Length panel is not longer than training paths")
    support={}
    for panel in sorted(panels):
        count=0
        for row in test:
            h,w=row["height"],row["width"]
            ratio=max(1,s.c["data"]["min_side"]/min(h,w));h,w=round(h*ratio),round(w*ratio)
            geometries=[codec.planned_geometry(h,w,c["stages"]) for c in s.c["measurement"]["conditions"] if c["panel"]==panel]
            count+=any(a*b>=65536 for a,b in geometries)
        support[panel]=count
        if count<2:raise Blocked("Planned geometry leaves fewer than two count-evaluable groups in "+panel+"; revise design before labels, keeping refusals in operating denominators")
    proof={"time":now(),"config":digest(s.c),"implementation":s.state["implementation"],
           "manifest":file_hash(s.root/"data_manifest.json"),"models":file_hash(s.root/"calibrated_models.json"),
           "selection":file_hash(s.root/"selection.json"),"sample_plan":file_hash(s.root/"sample_plan.json"),
           "acceptance":{"exact":"K=0","tau":.125,"guard_refusal":"known rejection; K missing, operating rate 0"},
           "test_groups":len(test),"planned_count_evaluable_groups":support,"claim_scope":s.c["design"]["source_scope"],
           "hypothesis_rule":"H_BP primary paired NLL CI lower > delta; H_BT BOTH transfer panels same gain AND acceptance MAE upper <= bound",
           "no_test_labels_opened":True}
    write_json(s.root/"freeze.json",proof)
    s.finish("freeze",["freeze.json"],test_readiness="READY_FOR_TEST")

def forecast(s):
    s.require("freeze")
    if s.state["test_accessed"]:raise Blocked("Cannot regenerate forecasts after test labels opened")
    models=read_json(s.root/"calibrated_models.json")
    sources=[r for r in read_json(s.root/"data_manifest.json") if r["split"]=="test"]
    folder=s.root/"forecasts";folder.mkdir(exist_ok=True)
    compute_log=s.root/"forecast_compute_batches.jsonl"
    rows=[];pending=[];batch_reports=read_rows(compute_log) if compute_log.is_file() else []
    source_batch=s.c.get("compute",{}).get("forecast_source_batch",1)
    def flush():
        if not pending:return
        flat=[(model_row,output) for source,part in pending for model_row,output in part]
        inputs=[pair[0] for pair in flat]
        for name,model in models.items():
            start=time.perf_counter();pmfs,compute_report=predict_pmfs(model,inputs,s.c.get("compute"));elapsed=time.perf_counter()-start
            record={"model":name,"sources":len(pending),"source_ids":[src["image_id"] for src,part in pending],"time":now(),**compute_report}
            append_json(compute_log,record);batch_reports.append(record)
            for (_,output),pmf in zip(flat,pmfs):
                events=acceptance(pmf) if output["supported"] else {"p_exact":0.,"p_tau":0.}
                output["models"][name]={"pmf":pmf.tolist(),**events,"inference_s":elapsed/len(flat),
                     "inference_timing":"amortized full numeric batch including CPU tensor assembly/transfers",
                     "inference_batch_rows":len(flat),"inference_device":compute_report["actual_device"]}
        for source,part in pending:
            outputs=[output for model_row,output in part]
            write_json(folder/(source["image_id"]+".json"),{"freeze":file_hash(s.root/"freeze.json"),"rows":outputs,"rows_hash":digest(outputs)})
            rows.extend(outputs)
        pending.clear()
    for source in progress(sources,"test input-only forecasts before channel labels"):
        chunk=folder/(source["image_id"]+".json")
        if chunk.exists():
            cached=read_json(chunk)
            if cached["freeze"]!=file_hash(s.root/"freeze.json") or digest(cached["rows"])!=cached["rows_hash"]:raise ValueError("Forecast checkpoint changed")
            rows.extend(cached["rows"]);continue
        start_io=time.perf_counter();image=s._verify_image(source);io=time.perf_counter()-start_io;part=[]
        for variant in s.c["measurement"]["variants"]:
            for payload in s.c["measurement"]["payloads"]:
                start=time.perf_counter();_,trace=codec.encode(image,codec.PAYLOADS[payload],s.c["measurement"]["delta"],variant["kind"],variant["swap"])
                encode_s=time.perf_counter()-start
                for cond in s.c["measurement"]["conditions"]:
                    start=time.perf_counter();f=codec.feature_dict(trace,variant,len(codec.PAYLOADS[payload]),s.c["measurement"]["delta"],cond["stages"])
                    h,w=codec.planned_geometry(*image.shape[:2],cond["stages"])
                    feature_s=time.perf_counter()-start
                    row={"variant":variant["id"],"payload_id":payload,"condition_id":cond["id"],"L":len(codec.PAYLOADS[payload]),"features":f}
                    output={"trial_id":digest({"image":source["image_id"],"variant":variant["id"],"payload":payload,"condition":cond["id"],"role":"public_mark"})[:32],
                            "image_id":source["image_id"],"group_id":source["group_id"],"panel":cond["panel"],"supported":h*w>=65536,
                            "feature_hash":digest(f),"image_read_s":io,"encode_and_features_s":encode_s,"feature_assembly_s":feature_s,"models":{}}
                    part.append((row,output))
        pending.append((source,part))
        if len(pending)>=source_batch:flush()
    flush()
    save_rows(s.root/"forecasts.jsonl",rows)
    write_json(s.root/"forecast_compute.json",{"batches":batch_reports,"source_batch":source_batch,
                  "images_on_gpu":False,"scope":"all numeric batch attempts, including resumed/repeated packets; image codec remains CPU reference"})
    append_json(s.root/"data_access.jsonl",{"time":now(),"split":"test","access":"carrier_and_pretransmission_features_only","groups":len(sources)})
    s.finish("forecast",["forecasts.jsonl","forecast_compute.json","forecast_compute_batches.jsonl"])

def _interval(s,values):
    return paired_interval(values,s.c["statistics"]["seed"],s.c["statistics"]["bootstrap_reps"],s.c["statistics"]["confidence"])

def evaluate(s):
    s.require("measure-test","forecast","freeze")
    rows=read_rows(s.root/"test_rows.jsonl");forecasts={r["trial_id"]:r for r in read_rows(s.root/"forecasts.jsonl")}
    results={};d=s.c["design"];joined=[]
    for r in progress([r for r in rows if r["role"]=="public_mark"],"join frozen predictions to labels"):
        f=forecasts.pop(r["trial_id"])
        if f["feature_hash"]!=digest(r["features"]) or f["supported"]!=r["decoder_supported"]:raise ValueError("Forecast/label feature or support mismatch")
        z=dict(r,predictions={})
        for name,pred in f["models"].items():
            cdf=np.cumsum(pred["pmf"]);lo=int(np.searchsorted(cdf,.025));hi=int(np.searchsorted(cdf,.975))
            z["predictions"][name]={"nll":None if r["K"] is None else -math.log(max(pred["pmf"][r["K"]],1e-300)),
                 "p_exact":pred["p_exact"],"p_tau":pred["p_tau"],
                 "brier_exact":(pred["p_exact"]-r["exact"])**2,"brier_tau":(pred["p_tau"]-r["tau_accept"])**2,
                 "central95_count_interval":[lo,hi],"covered":None if r["K"] is None else int(lo<=r["K"]<=hi)}
        joined.append(z)
    if forecasts:raise ValueError("Unused forecast rows")
    for panel in ("primary","transfer_order","transfer_length"):
        selected=[r for r in joined if r["panel"]==panel];groups=defaultdict(list)
        for r in selected:
            if r["K"] is not None:groups[r["group_id"]].append(r["predictions"]["baseline"]["nll"]-r["predictions"]["proposed"]["nll"])
        if len(groups)<2:raise Blocked("Insufficient decoder-supported groups for "+panel)
        ci=_interval(s,[np.mean(v) for v in groups.values()])
        # Fixed condition cells; estimate |mean predicted probability - event frequency|.
        # Resample independent image groups, retaining all paired conditions in each group.
        keys=sorted({(r["variant"],r["payload_id"],r["condition_id"]) for r in selected})
        gids=sorted({r["group_id"] for r in selected});gi={g:i for i,g in enumerate(gids)};ki={k:i for i,k in enumerate(keys)}
        residual=np.full((len(gids),len(keys),2),np.nan)
        for r in selected:
            residual[gi[r["group_id"]],ki[(r["variant"],r["payload_id"],r["condition_id"])]]=[
                r["predictions"]["proposed"]["p_exact"]-r["exact"],r["predictions"]["proposed"]["p_tau"]-r["tau_accept"]]
        if not np.isfinite(residual).all():raise ValueError("Missing operating cell in calibration error estimate")
        def mae(v):return float(np.mean(np.abs(v.mean(axis=0))))
        estimate=mae(residual);rng=np.random.default_rng(s.c["statistics"]["seed"]);boots=[]
        for _ in progress(range(s.c["statistics"]["bootstrap_reps"]),"group resampling "+panel):
            boots.append(mae(residual[rng.integers(0,len(gids),len(gids))]))
        tail=(1-s.c["statistics"]["confidence"])/2
        mae_ci={"estimate":estimate,"low":float(np.quantile(boots,tail)),"high":float(np.quantile(boots,1-tail)),"fixed_cells":len(keys)*2}
        gain=ci["low"]>d["delta_nll"]
        support=gain and (panel=="primary" or mae_ci["high"]<=d["acceptance_mae"])
        verdict="SUPPORTED" if support else ("NOT_SUPPORTED" if ci["high"]<=d["delta_nll"] or (panel!="primary" and mae_ci["low"]>d["acceptance_mae"]) else "INCONCLUSIVE")
        results[panel]={"paired_nll_gain":ci,"acceptance_probability_mae":mae_ci,"verdict":verdict,
                        "decoder_supported_rate":sum(r["K"] is not None for r in selected)/len(selected),
                        "operating_groups":len(gids),"rows":len(selected),"delta_nll":d["delta_nll"]}
        metrics={};reliability=[]
        for name in ("baseline","proposed"):
            grouped=defaultdict(list)
            for r in selected:grouped[r["group_id"]].append(r)
            metrics[name]={}
            for metric in ("nll","brier_exact","brier_tau","covered"):
                values=[]
                for items in grouped.values():
                    v=[r["predictions"][name][metric] for r in items if r["predictions"][name][metric] is not None]
                    if v:values.append(np.mean(v))
                metrics[name][metric]=float(np.mean(values)) if values else None
            for event,label in (("p_exact","exact"),("p_tau","tau_accept")):
                for bin_id in range(10):
                    cell=[r for r in selected if min(9,int(r["predictions"][name][event]*10))==bin_id]
                    reliability.append({"model":name,"event":event,"bin":bin_id,"rows":len(cell),
                        "mean_prediction":float(np.mean([r["predictions"][name][event] for r in cell])) if cell else None,
                        "event_rate":float(np.mean([r[label] for r in cell])) if cell else None})
        results[panel]["model_metrics"]=metrics;results[panel]["fixed_bin_reliability_descriptive"]=reliability
    verdicts={"H_BP":results["primary"]["verdict"],"H_BT":"SUPPORTED" if all(results[p]["verdict"]=="SUPPORTED" for p in ("transfer_order","transfer_length")) else ("NOT_SUPPORTED" if any(results[p]["verdict"]=="NOT_SUPPORTED" for p in ("transfer_order","transfer_length")) else "INCONCLUSIVE")}
    # Intersection-union H_BT: both panels and criteria must pass, no best-panel selection.
    operating=defaultdict(list)
    for r in rows:operating[(r["variant"],r["payload_id"],r["condition_id"],r["role"])].append(r)
    rates=[]
    for key,items in operating.items():
        rates.append({"variant":key[0],"payload":key[1],"condition":key[2],"role":key[3],
                      "metric":"TPR/public deterministic insertion success" if key[3]=="public_mark" else "natural FPR",
                      "exact":exact_ci(sum(r["exact"] for r in items),len(items),s.c["statistics"]["confidence"]),
                      "tau":exact_ci(sum(r["tau_accept"] for r in items),len(items),s.c["statistics"]["confidence"]),
                      "guard_refusals":sum(r["K"] is None for r in items)})
    save_rows(s.root/"scored_test_rows.jsonl",joined)
    write_json(s.root/"evaluation.json",{"panels":results,"hypotheses":verdicts,"operating_rates":rates,
        "family":"two hypotheses each at confidence .975; H_BT intersection-union, exploratory cells unadjusted",
        "scope":"fresh images in specified source corpora, predeclared watermark-component and channel paths only"})
    s.finish("evaluate",["scored_test_rows.jsonl","evaluation.json"],hypotheses=verdicts)

def audit(s):
    s.require("mathcheck","source-audit","prepare")
    problems=[];checked=0
    manifest=read_json(s.root/"data_manifest.json");byimage={r["image_id"]:r for r in manifest}
    if len({r["group_id"] for r in manifest})!=len(manifest):problems.append("split-group overlap")
    for split in ("pilot","fit","validation","calibration","test"):
        path=s.root/(split+"_rows.jsonl")
        if not path.is_file():continue
        rows=read_rows(path);seen=set();actual=Counter()
        for r in progress(rows,"independent raw audit "+split):
            checked+=1
            if r["trial_id"] in seen:problems.append("duplicate trial")
            seen.add(r["trial_id"]);actual[r["image_id"]]+=1
            source=byimage[r["image_id"]]
            if source["split"]!=split or source["group_id"]!=r["group_id"]:problems.append("manifest join mismatch")
            if r["K"] is None:
                if r["decoder_supported"] or r["status"]!="guard_rejected" or r["exact"] or r["tau_accept"]:problems.append("invalid guard row")
            else:
                bits=np.asarray(r["bits"],int);ref=codec.PAYLOADS[r["payload_id"]]
                errors=(bits!=ref).astype(int)
                if len(bits)!=r["L"] or errors.tolist()!=r["errors"] or int(errors.sum())!=r["K"]:problems.append("raw bit/count mismatch")
                if int(not errors.any())!=r["exact"] or int(errors.sum()<=math.floor(.125*len(bits)))!=r["tau_accept"]:problems.append("acceptance arithmetic mismatch")
            forbidden={"K","bits","errors","decoded","postchannel_argmax_same","postchannel_block_error"}&set(r["features"])
            if forbidden:problems.append("label in feature schema")
        conds=[c for c in s.c["measurement"]["conditions"] if split=="test" or c["panel"]=="primary"]
        expected=len(conds)*len(s.c["measurement"]["variants"])*len(s.c["measurement"]["payloads"])*2
        if any(actual[r["image_id"]]!=expected for r in manifest if r["split"]==split):problems.append("missing image-condition-role cell in "+split)
    if s.state["test_accessed"] and s.state["stages"].get("freeze",{}).get("status")!="TECH_PASS":problems.append("test before freeze")
    report={"technical_audit":"PASS" if not problems else "FAIL","raw_rows_recomputed":checked,"problems":sorted(set(problems)),
            "confirmatory_status":s.state["test_readiness"],"hypotheses":s.state["hypotheses"],
            "missing_required_stages":[k for k in ("plan","select","calibrate","freeze","forecast","measure-test","evaluate","security","cost","mechanisms") if s.state["stages"].get(k,{}).get("status")!="TECH_PASS"],
            "meaning":"No detected technical contradiction is distinct from completed independent experimentation"}
    write_json(s.root/"audit.json",report)
    if problems:raise ValueError("Independent audit failed: "+str(report["problems"]))
    s.finish("audit",["audit.json"])

def report(s):
    audit(s)
    title="Auditing and Predicting Failures of Public Fixed-Payload Stable Diffusion Watermarks"
    evalpath=s.root/"evaluation.json"
    ev=read_json(evalpath) if evalpath.exists() else None
    missing=read_json(s.root/"audit.json")["missing_required_stages"]
    text=f"# {title}\n\n## Abstract\n\nThis study specifies an implementation-aware count model and separates public watermark reliability from authenticity. "
    text+=("Independent evaluation has not been completed. No confirmatory effect estimates are reported.\n" if ev is None else "Frozen independent evaluation is recorded in evaluation.json; all numerical claims below follow that artifact.\n")
    text+="\n## Introduction\n\nThe original ICISC audit supplies development evidence. Newly collected, deduplicated source groups are required for calibration and independent testing. Publication venue does not itself establish or refute technical contribution.\n"
    text+="\n## Experiments\n\nH_BP tests a group-paired count-NLL improvement over the strongest development-selected baseline. H_BT requires both unseen-order and longer-path panels to retain improvement and meet the locked acceptance-probability error bound. RQ_M, RQ_S and RQ_C address mechanisms, operational security scenarios and full computation cost. Decoder guard refusals remain in operational denominators and have no fabricated error count.\n"
    if ev:
        for key,value in ev["hypotheses"].items():text+=f"\n- {key}: {value}.\n"
        for panel,value in ev["panels"].items():
            c=value["paired_nll_gain"];text+=f"\n{panel}: NLL gain {c['estimate']:.6f}, CI [{c['low']:.6f}, {c['high']:.6f}], {value['verdict']}.\n"
    else:text+="\nThe available results are technical checks and development measurements only. Assumed outcomes are not observations.\n"
    text+="\n## Conclusion\n\n"+("The claims must follow supported scopes and valid negative or inconclusive outcomes. " if ev else "The design is implementable, but the new scientific hypotheses remain unevaluated. ")
    text+="Outstanding stages: "+(", ".join(missing) if missing else "none")+".\n"
    (s.root/"paper_B_measured_draft.md").write_text(text,encoding="utf-8")
    s.finish("report",["paper_B_measured_draft.md"])
