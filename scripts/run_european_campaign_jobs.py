"""Run shared-pipeline jobs with explicit code snapshots and per-language receipts."""
import argparse,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dala.batch_pipeline import atomic_json
from dala.common_pile import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('launch',type=Path);args=p.parse_args()
    launch=json.loads(args.launch.read_text());logs=Path(launch['language_logs']);logs.mkdir(parents=True,exist_ok=True)
    def run(record):
        if record.get('completed') or record.get('external'):return
        profile=Path(record['profile'])
        if sha256_file(profile)!=record['profile_sha256']:raise ValueError('Profile changed before launch')
        actual={str(p.relative_to(Path(record['cwd'])/'dala')):sha256_file(p) for p in (Path(record['cwd'])/'dala').rglob('*.py')}
        if actual!=record['code_sha256']:raise ValueError('Code changed before launch')
        command=[sys.executable,'-u','-m','scripts.run_european_pilots','--worker','--languages',record['language'],
                 '--run-id',record['run_id'],'--max-documents','0','--profile-root',str(profile.parent.parent)]
        log=logs/(record['language']+'.log')
        if log.exists():raise FileExistsError(log)
        with log.open('w') as output:
            result=subprocess.run(command,cwd=record['cwd'],stdout=output,stderr=subprocess.STDOUT)
        status=dict(language=record['language'],exit_code=result.returncode,log=str(log),output=record['output'])
        atomic_json(logs/(record['language']+'-status.json'),status);print(json.dumps(status),flush=True)
    with ThreadPoolExecutor(max_workers=sum(not r.get('completed') and not r.get('external') for r in launch['languages'])) as pool:
        list(pool.map(run,launch['languages']))

if __name__=='__main__':main()
