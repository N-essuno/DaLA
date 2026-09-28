"""Prepare or run resumable separate candidate builds; never publish automatically."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import hashlib

SETTINGS={'pl':(64,8),'sv':(64,8),'nb':(64,2),'nn':(64,2),'fo':(64,32),'is':(64,2)}


def prepare(language,run_id,tools_dir,parser_workers=None):
    root=Path('config/languages')
    profile=json.loads((root/f'{language}.json').read_text())
    processes,batch=SETTINGS[language]
    if parser_workers is not None:
        if parser_workers < 1:raise ValueError("parser workers must be positive")
        processes=parser_workers
    profile['build']=dict(parser_processes=processes,document_batch_size=batch,max_task_chars=30000,
                          checkpoint_dir=f'la_output/{language}_{run_id}_checkpoints',allow_shortfall=True)
    profile['build']['prestart_parser_workers']=True
    profile['build']['worker_pool_sha256']=hashlib.sha256(Path('scripts/worker_pool.py').read_bytes()).hexdigest()
    target=profile['release_policy']['target']['target_pairs']
    if target:
        profile['build'].update(target_pairs=target,split_targets={'train':383144,'validation':47893,'test':47893})
    if profile['checker'].get('source_languagetool'):
        profile['checker'].update(tools_dir=tools_dir,workers=32)
        profile['build']['persistent_screening_workers']=True
        profile['build']['screening_pool_sha256']=hashlib.sha256(Path('scripts/screening_pool.py').read_bytes()).hexdigest()
    path=root/f'{language}_{run_id}.json'
    encoded=json.dumps(profile,ensure_ascii=False,indent=2)+'\n'
    if path.exists():
        if json.loads(path.read_text())!=profile:raise ValueError(f'Existing run profile differs: {path}; choose a new run ID')
        return path
    path.write_text(encoded)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--languages',nargs='+',choices=list(SETTINGS),default=list(SETTINGS))
    p.add_argument('--run-id',default='scale_v1')
    p.add_argument('--tools-dir',default='la_output/tools')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--parser-workers',type=int,help='Override parser process count for this run')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--transport-retries',type=int,default=2)
    a=p.parse_args()
    if a.worker:
        from dala.pipeline import build
        from dala.profiles import load_profile
        language=a.languages[0]
        profile=load_profile(f'config/languages/{language}_{a.run_id}.json')
        if profile['build'].get('prestart_parser_workers'):
            if hashlib.sha256(Path('scripts/worker_pool.py').read_bytes()).hexdigest()!=profile['build']['worker_pool_sha256']:
                raise ValueError('Pinned worker-pool implementation changed')
            from scripts.worker_pool import PrestartedProcessPool
            from dala import batch_pipeline
            batch_pipeline.ProcessPoolExecutor=PrestartedProcessPool
        close_pool=None
        if profile['build'].get('persistent_screening_workers'):
            if hashlib.sha256(Path('scripts/screening_pool.py').read_bytes()).hexdigest()!=profile['build']['screening_pool_sha256']:
                raise ValueError('Pinned screening-pool implementation changed')
            from scripts.screening_pool import PersistentScreeningPool, close_screening_pools
            from dala import batch_pipeline
            batch_pipeline.ThreadPoolExecutor=PersistentScreeningPool
            close_pool=close_screening_pools
        try:
            build(profile,output_dir=f'la_output/{language}_dynaword_{a.run_id}',seed=4242,max_errors=1)
        finally:
            if close_pool:close_pool()
        return
    for language in a.languages:prepare(language,a.run_id,a.tools_dir,a.parser_workers)
    if a.prepare_only:return
    root=Path('wiki/artifacts/six-language-expansion')/a.run_id;root.mkdir(parents=True,exist_ok=True)
    def run(language):
        logfile=root/f'{language}.log'
        for attempt in range(a.transport_retries+1):
            with logfile.open('a') as log:
                offset=log.tell()
                result=subprocess.run([sys.executable,'-u','-m','scripts.run_six_language_scale','--worker',
                                       '--languages',language,'--run-id',a.run_id],stdout=log,stderr=subprocess.STDOUT)
            with logfile.open() as log:
                log.seek(offset);tail=log.read()[-12000:]
            transport=any(error in tail for error in ['requests.exceptions.ConnectionError','requests.exceptions.ReadTimeout'])
            if result.returncode==0 or not transport or attempt==a.transport_retries:break
            print(f'{language}: local checker transport failed; resuming verified checkpoints, retry {attempt+1}',flush=True)
            time.sleep(5)
        record=dict(language=language,exit_code=result.returncode,log=str(logfile),output=f'la_output/{language}_dynaword_{a.run_id}',
                    status='candidate_build_complete_requires_final_audit' if result.returncode==0 else 'failed_checkpoints_preserved')
        (root/f'{language}-status.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        return record
    with ThreadPoolExecutor(max_workers=len(a.languages)) as pool:results=list(pool.map(run,a.languages))
    if any(r['exit_code'] for r in results):raise SystemExit(1)


if __name__=='__main__':main()
