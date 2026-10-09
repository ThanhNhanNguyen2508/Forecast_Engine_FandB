"""Direct typed procurement CLI using the same public M5/M6/export services."""
from pathlib import Path
import argparse,json,sys
from shelfcash_forecast.optimization.planning_service import run_typed_planning

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    import shelfcash_forecast,shelfcash_pipeline,shelfcash_preprocess
    print(json.dumps({'python':sys.executable,'forecast':shelfcash_forecast.__file__,'pipeline':shelfcash_pipeline.__file__,'preprocess':shelfcash_preprocess.__file__}),flush=True)
    try:payload=json.loads(args.request.read_text(encoding='utf-8-sig'))
    except (ValueError,OSError) as exc:
        print(json.dumps({'status':'INVALID_INPUT','diagnostics':[{'code':'REQUEST_JSON_REQUIRED','field_paths':['$'],'message':str(exc)}]}));return 2
    run=run_typed_planning(payload,destination=args.output)
    result=run.result if hasattr(run,'result') else run
    print(result.model_dump_json(exclude={'evaluations'}),flush=True)
    return 0 if result.technical_outcome not in {'INVALID_INPUT','SOLVER_ERROR'} else 2

if __name__=='__main__':raise SystemExit(main())
