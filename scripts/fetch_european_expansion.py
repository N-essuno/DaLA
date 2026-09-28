"""Fetch exactly pinned public resource bytes; no mutable-head discovery or GPU use."""
import argparse
import json
from pathlib import Path

import requests
from dala.common_pile import sha256_file
from scripts.prepare_european_sources import write


def fetch(url,path,digest):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.download')
        with requests.get(url,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            with temporary.open('wb') as out:
                for chunk in response.iter_content(1024*1024):out.write(chunk)
        if sha256_file(temporary)!=digest:raise ValueError('Downloaded resource checksum mismatch')
        temporary.replace(path)
    if sha256_file(path)!=digest:raise ValueError('Existing resource checksum mismatch')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lock',type=Path,default=Path('config/european-expansion/resource-lock.json'));p.add_argument('--output',type=Path,default=Path('la_output/resources/european-expansion'));a=p.parse_args()
    lock=json.loads(a.lock.read_text());root=a.output.resolve()
    for code,receipt in lock['unimorph'].items():
        folder=root/'unimorph'/code
        for record in receipt['files']:fetch(record['url'],folder/record['file'],record['sha256'])
        if not (folder/'receipt.json').exists():write(folder/'receipt.json',receipt)
        if not (folder/'pin.json').exists():write(folder/'pin.json',dict(revision=receipt['revision']))
        print(code,'verified',flush=True)
    from huggingface_hub import hf_hub_download
    for language,receipt in lock['wikipedia'].items():
        path=hf_hub_download(repo_id=receipt['repo_id'],repo_type='dataset',revision=receipt['revision'],filename=receipt['file'])
        if sha256_file(path)!=receipt['sha256']:raise ValueError('Wikipedia checksum mismatch')
        target=root/'sources'/(language+'-wikipedia.json')
        if not target.exists():write(target,dict(receipt,path=path))
        print(language,'Wikipedia verified',flush=True)
    for language,receipt in lock.get('nominal',{}).items():
        folder=root/'nominal'/language
        records=[]
        for record in receipt['files']:
            path=folder/record['file'];fetch(record['url'],path,record['sha256'])
            records.append(dict(path=str(path),url=record['url'],sha256=record['sha256']))
        if not (folder/'receipt.json').exists():write(folder/'receipt.json',dict(receipt,files=records))
        print(language,'nominal lexicon verified',flush=True)
    recipe=lock['europarl'];fetch(recipe['url'],root/recipe['file'],recipe['sha256'])
    print('Europarl verified',flush=True)


if __name__=='__main__':main()
