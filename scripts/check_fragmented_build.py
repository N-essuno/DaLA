"""Compare real ordinary, fragmented and resumed builds on one Nynorsk document."""
import json
import argparse
from pathlib import Path
from dala.profiles import load_profile
from dala.morphology_check import MorphologyCheck
from dala.pair_pipeline import _build
from dala.batch_pipeline import build_checkpointed
from dala.validate_dataset import validate


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run-id',default='')
    cli.add_argument('--prestart-workers',type=int)
    args=cli.parse_args()
    suffix='_'+args.run_id if args.run_id else ''
    profile=load_profile('nn')
    if args.prestart_workers:
        from scripts.worker_pool import PrestartedProcessPool
        from dala import batch_pipeline
        batch_pipeline.ProcessPoolExecutor=PrestartedProcessPool
    profile['build']=dict(parser_processes=args.prestart_workers or 2,document_batch_size=1,max_task_chars=10000,
                          checkpoint_dir=f'la_output/nn_fragment_equivalence{suffix}_checkpoints')
    path=Path(f'config/languages/nn_fragment_equivalence{suffix}.json')
    path.write_text(json.dumps({k:v for k,v in profile.items() if k!='_path'},ensure_ascii=False,indent=2)+'\n')
    profile=load_profile(path);checker=MorphologyCheck(profile)
    outputs=[]
    try:
        for name,build in [('ordinary',_build),('fragmented',build_checkpointed),('resumed',build_checkpointed)]:
            output=Path('la_output/nn_fragment_equivalence'+suffix+'_'+name)
            if not output.exists():build(output_dir=output,max_documents=1,max_errors=1,seed=4242,profile=profile,checker=checker)
            rows=[json.loads(s) for f in sorted(output.glob('*/pairs.jsonl')) for s in f.open()]
            outputs.append((output,sorted(rows,key=lambda r:r['pair_id']),validate(output)))
        assert outputs[0][1]==outputs[1][1]==outputs[2][1], 'Pair/evidence/offset mismatch'
        record=dict(passed=True,pairs=len(outputs[0][1]),checks=['identical canonical pairs','identical source offsets','identical checker evidence','identical resumed output'],
                    outputs={str(p):v for p,_,v in outputs})
        Path('wiki/artifacts/six-language-expansion/fragment-equivalence'+suffix+'.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    finally:checker.close()


if __name__=='__main__':main()
