"""Run separate packs through dala.pipeline.build; no alternate generation path."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--languages',nargs='+')
    parser.add_argument('--run-id',default='pilot_v1')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--profile-root',type=Path,help='Directory containing CODE/profile.json; defaults to existing pilot profiles')
    parser.add_argument('--max-documents',type=int,default=2000,help='Document limit; 0 uses the complete prepared source and enables configured split quotas')
    args=parser.parse_args()
    languages=args.languages or [p.stem for p in sorted(Path('config/european').glob('*.json')) if not p.stem.endswith(('-sources','-exclusions'))]
    root=Path('wiki/artifacts/european-pilots')/args.run_id;root.mkdir(parents=True,exist_ok=True)
    if args.worker:
        from dala.pipeline import build
        from dala.profiles import load_profile
        from dala.common_pile import sha256_file
        profile=load_profile(str(args.profile_root/languages[0]/'profile.json') if args.profile_root else languages[0])
        settings=profile.get('build', {})
        close_pool=None
        if settings.get('prestart_parser_workers'):
            if sha256_file('scripts/worker_pool.py')!=settings['worker_pool_sha256']:raise ValueError('Worker pool implementation changed')
            from scripts.worker_pool import PrestartedProcessPool
            from dala import batch_pipeline
            batch_pipeline.ProcessPoolExecutor=PrestartedProcessPool
        if settings.get('persistent_screening_workers'):
            if sha256_file('scripts/screening_pool.py')!=settings['screening_pool_sha256']:raise ValueError('Screening pool implementation changed')
            from scripts.screening_pool import PersistentScreeningPool, close_screening_pools
            from dala import batch_pipeline
            batch_pipeline.ThreadPoolExecutor=PersistentScreeningPool
            close_pool=close_screening_pools
        print(f"Starting {languages[0]}: {settings.get('parser_processes', 1)} CPU parser workers; document limit {args.max_documents or 'none'}",flush=True)
        try:
            build(profile,output_dir=f'la_output/european_pilots/{args.run_id}/{languages[0]}',max_documents=args.max_documents or None,seed=4242)
        finally:
            if close_pool:close_pool()
        return
    def run(language):
        log=root/f'{language}.log'
        with log.open('w') as output:
            command=[sys.executable,'-u','-m','scripts.run_european_pilots','--worker','--languages',language,'--run-id',args.run_id,'--max-documents',str(args.max_documents)]
            if args.profile_root: command.extend(['--profile-root',str(args.profile_root)])
            result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
        status=dict(language=language,exit_code=result.returncode,log=str(log),output=f'la_output/european_pilots/{args.run_id}/{language}')
        (root/f'{language}-status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
        return status
    with ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(run,languages))
    if any(r['exit_code'] for r in results):raise SystemExit(1)


if __name__=='__main__':main()
