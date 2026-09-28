"""Inventory expert-authored checker examples separately from observed corpora.

Only explicitly grammar/spelling-typed rules are included. These contextual
examples are review fixtures, never context-free word replacement mappings.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from urllib.parse import urlparse, unquote

from dala.common_pile import sha256_file


def extract(path,language,source_language,root):
    from lxml import etree
    dependencies = {}
    class LocalResolver(etree.Resolver):
        def resolve(self,url,pubid,context):
            parsed=urlparse(url)
            if parsed.scheme not in {'','file'}:raise ValueError('No network entity resolution')
            target=Path(unquote(parsed.path)).resolve()
            if not target.is_relative_to(root.parent.resolve()):raise ValueError('Entity outside pinned checker resources')
            dependencies[str(target)] = sha256_file(target)
            return self.resolve_filename(str(target),context)
    parser=etree.XMLParser(load_dtd=True,resolve_entities=True,no_network=True)
    parser.resolvers.add(LocalResolver())
    tree=etree.parse(str(path.resolve()),parser)
    if tree.getroot().get('lang')!=source_language:raise ValueError('Rule example language mismatch')
    rows=[];counts=Counter()
    for rule in tree.findall('.//rule'):
        chain=[rule,*rule.iterancestors()]
        kind=next((e.get('type') for e in chain if e.get('type')),None)
        if kind not in {'grammar','misspelling'}:
            counts['untyped_or_out_of_scope_rules']+=1;continue
        category=next((e for e in chain if e.tag=='category'),None)
        ids=[e.get('id') for e in reversed(chain) if e.get('id')]
        for example in rule.findall('example'):
            correction=example.get('correction')
            if correction is None:continue
            markers=list(example.iter('marker'))
            if len(markers)!=1 or len(list(example))!=1 or '|' in correction or not correction:
                counts['complex_or_alternative_example']+=1;continue
            marker=markers[0]
            wrong=''.join(marker.itertext())
            before=example.text or '';after=marker.tail or ''
            original=before+wrong+after;corrected=before+correction+after
            if original==corrected:continue
            rows.append(dict(language=language,source_language=source_language,rule_id='/'.join(ids),
                category=category.get('name') if category is not None else None,type=kind,
                incorrect=original,correct=corrected,error=wrong,correction=correction,
                status='expert_authored_contextual_fixture_not_observed_error_not_activated'))
    return rows,dict(counts),dependencies


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recipes',type=Path,default=Path('config/european-expansion/checker-evidence.json'));p.add_argument('--output',type=Path,default=Path('la_output/resources/european-expansion/checker-evidence'));a=p.parse_args()
    recipes=json.loads(a.recipes.read_text());root=Path(recipes['root'])
    summary={}
    for recipe in recipes['languages']:
        path=root/recipe['file']
        if sha256_file(path)!=recipe['sha256']:raise ValueError('Rule XML checksum mismatch')
        rows,counts,dependencies=extract(path,recipe['language'],recipe['source_language'],root)
        out=a.output/recipe['language'];out.mkdir(parents=True,exist_ok=True)
        (out/'fixtures.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        report=dict(language=recipe['language'],examples=len(rows),rules=len({r['rule_id'] for r in rows}),counts=counts,recipe=recipe,dependency_sha256=dependencies,
                    source='LanguageTool 6.6, LGPL-2.1-or-later; expert-authored examples, not naturally observed mistakes',
                    qualification='pt is a shared-standard rule source; pt-PT applicability still requires checking' if recipe['language']=='pt-PT' else 'No native review or automatic activation')
        (out/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');summary[recipe['language']]=report
        print(recipe['language'],len(rows),flush=True)
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
