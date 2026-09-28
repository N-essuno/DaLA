"""Apply recorded language-specific audit inputs to separate candidate profiles."""
import argparse
import json
from pathlib import Path

from scripts.prepare_european_sources import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,default=Path('config/european-expansion/audit-policy.json'))
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coverage-targets',type=Path)
    p.add_argument('--parser-workers',type=int)
    p.add_argument('--checkpoint-root',type=Path)
    p.add_argument('--review-directory',type=Path,action='append',default=[])
    args=p.parse_args()
    if args.parser_workers is not None and (args.parser_workers<1 or args.checkpoint_root is None):p.error('positive parser workers and a checkpoint root are required together')
    targets=json.loads(args.coverage_targets.read_text())['languages'] if args.coverage_targets else {}
    for policy in json.loads(args.policy.read_text())['languages']:
        language=policy['language'];out=args.output.resolve()/language
        if out.exists():raise FileExistsError(out)
        profile=json.loads((args.input/language/'profile.json').read_text())
        if profile['language']!=language:raise ValueError('Audit policy language mismatch')
        exclusions=set(policy['exclusions'])
        for directory in args.review_directory:
            path=directory/(language+'.json')
            if path.exists():
                review=json.loads(path.read_text())
                if review['language']!=language:raise ValueError('Review language mismatch')
                exclusions.update(review['source_sha256'])
        write(out/'exclusions.json',sorted(exclusions))
        profile['exclusions']=str(out/'exclusions.json')
        profile['curation']['source_risk_patterns']+=policy['source_risk_patterns']
        profile['morphology_options']['protect_capitalized_lemmas']=policy['protect_capitalized_lemmas']
        profile['checker']['source_languagetool']=policy['source_languagetool']
        profile['description']='CPU expanded pilot, agent source exclusions and stricter extraction guards; independent source checker where supported. Candidate only, no native validation.'
        if language in targets:profile['selection']['coverage_targets']=[r['family'] for r in targets[language]]
        if args.parser_workers:
            profile['parser_threads']=1
            profile['build']=dict(parser_processes=args.parser_workers,document_batch_size=10,max_task_chars=30000,checkpoint_dir=str((args.checkpoint_root/language).resolve()),allow_shortfall=True)
            profile['checker']['workers']=8
        write(out/'profile.json',profile)


if __name__=='__main__':main()
