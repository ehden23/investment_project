"""Run bundled prepared inputs from any working directory; no API credentials."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from team7_quant.config import load_config
from team7_quant.module8 import run_risk_evaluation, check_risk_inputs

if __name__ == "__main__":
    p=argparse.ArgumentParser(description='모듈 7·8 실행: 임의 비용/헤지 예산을 채우지 않습니다.')
    p.add_argument('--check', action='store_true')
    p.add_argument('--baseline-only', action='store_true')
    p.add_argument('--config', type=Path, default=ROOT/'config/base.json')
    p.add_argument('--prepared', type=Path, default=ROOT/'data/risk_prepared')
    p.add_argument('--baseline', type=Path, default=ROOT/'results/backtest')
    p.add_argument('--out', type=Path, default=ROOT/'results/module7_8')
    args=p.parse_args()
    try:
        cfg=load_config(args.config)
        for key in ['cost_file','cash_file']:
            value=cfg.get('risk',{}).get(key)
            if value and not Path(value).is_absolute():cfg['risk'][key]=str(ROOT/value)
        if args.check:
            print(json.dumps(check_risk_inputs(args.prepared,cfg,args.baseline),ensure_ascii=False,indent=2))
        else:
            summary=run_risk_evaluation(args.prepared,args.out,cfg,args.baseline,baseline_only=args.baseline_only)
            print(summary[['strategy_id','scenario','cagr','ann_vol','sharpe','mdd']].to_string(index=False))
            print(args.out/'report.html')
    except (ValueError,FileNotFoundError) as exc:
        print(str(exc),file=sys.stderr);sys.exit(2)
