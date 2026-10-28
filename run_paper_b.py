"""All commands echo progress and save a UTF-8 terminal transcript."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"revision_B"/"src"))
from paper_b.common import terminal_log,Blocked,read_json
from paper_b.study import Study
from paper_b import workflow,operations,acquire

def main():
    p=argparse.ArgumentParser(description="Paper B gated research pipeline; technical pass is not hypothesis support")
    p.add_argument("--config",required=True)
    p.add_argument("command",choices=["init","status","environment","gpu-profile","source-audit","mathcheck","import-legacy","acquire","prepare","measure","pilot","plan","select","calibrate","freeze","forecast","evaluate","security","cost","mechanisms","audit","report"])
    p.add_argument("--split",choices=["pilot","fit","validation","calibration","test"])
    p.add_argument("--parts",type=int,nargs="+",default=[2,3])
    a=p.parse_args();s=Study(a.config,create=a.command=="init")
    if a.command=="init":return 0
    if a.command=="status":
        import json
        print(json.dumps(s.state,ensure_ascii=False,indent=2));return 0
    stage="measure-"+str(a.split) if a.command=="measure" else a.command
    with terminal_log(s.root,stage):
        try:
            if stage not in ("audit","report") and s.state["stages"].get(stage,{}).get("status")=="TECH_PASS":
                print("[ALREADY COMMITTED] "+stage+"; use a new study_id for changed parameters",flush=True)
                return 0
            if a.command=="measure":
                if a.split is None:p.error("measure requires --split")
                s.measure(a.split)
            elif a.command=="acquire":acquire.acquire(s,a.parts)
            elif hasattr(s,a.command.replace("-","_")):getattr(s,a.command.replace("-","_"))()
            elif hasattr(workflow,a.command):getattr(workflow,a.command)(s)
            else:getattr(operations,a.command.replace("-","_"))(s)
        except Exception as exc:
            s.failure(stage,exc)
            if isinstance(exc,Blocked):return 3
            import traceback
            traceback.print_exc();return 1
    return 0

if __name__=="__main__":raise SystemExit(main())
