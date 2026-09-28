"""Run bounded, separately exported candidate builds; never publish or certify them."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def run(language, documents, run_id, all_sources=False):
    from dala.pipeline import build
    from dala.profiles import load_profile
    profile = load_profile(language)
    # First candidate source is deliberately audited before broadening the pool.
    source_config = json.loads((Path(profile['_path']).parent / profile['sources_config']).read_text())
    if not all_sources: source_config['sources'] = source_config['sources'][:1]
    # Uniform raw-row sampling before Python materialization keeps bounded
    # pilots representative without loading every full legal document.
    for source in source_config['sources']:
        source['pilot_sample_rows']=max(100,documents*3)
        source['pilot_sample_seed']=4242
    folder = Path('wiki/artifacts/six-language-expansion')
    path = (folder / f'{language}-{run_id}-sources.json').resolve()
    path.write_text(json.dumps(source_config, indent=2) + '\n')
    destination = f'la_output/{language}_dynaword_{run_id}'
    return build(profile, output_dir=destination, sources_path=path, max_documents=documents,
                 max_errors=1, seed=4242)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--languages',nargs='+',default=['pl','sv','nb','nn','fo','is'])
    parser.add_argument('--documents',type=int,default=20)
    parser.add_argument('--run-id',default='pilot_v1')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--all-sources',action='store_true')
    args=parser.parse_args()
    if args.worker:
        run(args.languages[0],args.documents,args.run_id,args.all_sources);return
    root=Path('wiki/artifacts/six-language-expansion')
    def worker(language):
        path=root/f'{language}-{args.run_id}.log'
        with path.open('w') as log:
            result=subprocess.run([sys.executable,'-u','-m','scripts.run_six_language_pilots','--worker','--languages',language,
                                   '--documents',str(args.documents),'--run-id',args.run_id]+(['--all-sources'] if args.all_sources else []),stdout=log,stderr=subprocess.STDOUT)
        record={'language':language,'exit_code':result.returncode,'log':str(path),
                'output':f'la_output/{language}_dynaword_{args.run_id}'}
        (root/f'{language}-{args.run_id}-status.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        return record
    with ThreadPoolExecutor(max_workers=6) as pool: results=list(pool.map(worker,args.languages))
    if any(r['exit_code'] for r in results):raise SystemExit(1)


if __name__=='__main__':main()
