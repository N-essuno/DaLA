"""Publish the explicitly authorized, locally validated Dutch HF package."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

ROOT=Path('export-upload/dala-dutch-dynaword')
REPO='schneiderkamplab/dala-dutch-dynaword'
EVIDENCE=Path('wiki/artifacts')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    m=json.loads((ROOT/'metadata/manifest.json').read_text())
    v=json.loads((ROOT/'metadata/validation.json').read_text())
    assert m['repo_id']==REPO and v['status']=='passed'
    paths=set(m['files'])|{'metadata/manifest.json','metadata/validation.json'}
    for name,record in m['files'].items():
        p=ROOT/name
        assert p.stat().st_size==record['bytes'] and digest(p)==record['sha256'],name
    loader={}
    for task in ['acceptability','correction']:
        ds=load_dataset(str(ROOT.resolve()),name=task)
        counts={s:len(ds[s]) for s in ds}
        assert counts=={s:r['rows_per_task'] for s,r in m['splits'].items()}
        loader[task]=counts
        print('Full local loader passed:',task,counts,flush=True)
        del ds
    (EVIDENCE/'dutch-hf-loader-validation.json').write_text(json.dumps(dict(status='passed',configurations=loader),indent=2)+'\n')
    api=HfApi()
    try:
        api.dataset_info(REPO)
    except RepositoryNotFoundError:
        api.create_repo(REPO,repo_type='dataset',private=False)
    else:
        raise RuntimeError('Repository already exists; refusing to overwrite without inspecting it')
    commit=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(ROOT),allow_patterns=sorted(paths),commit_message='Publish audited DaLA Dutch DynaWord grammar and spelling dataset')
    revision=commit.oid
    (EVIDENCE/'dutch-hf-upload-pending-verification.json').write_text(json.dumps(dict(repo_id=REPO,revision=revision,url=f'https://huggingface.co/datasets/{REPO}'),indent=2)+'\n')
    info=api.dataset_info(REPO,revision=revision,files_metadata=True)
    remote={s.rfilename:s for s in info.siblings}
    assert set(remote)-{'.gitattributes'}==paths
    for name in paths:
        item=remote[name];p=ROOT/name
        assert item.size==p.stat().st_size,name
        if item.lfs:
            assert item.lfs.sha256==digest(p),name
        else:
            blob=hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0'+p.read_bytes()).hexdigest()
            assert item.blob_id==blob,name
    for task in ['acceptability','correction']:
        ds=load_dataset(REPO,task,revision=revision,streaming=True)
        assert set(ds)==set(m['splits'])
        for split in ds:
            row=next(iter(ds[split]));assert row['metadata']['task']==task and row['metadata']['split']==split
            assert [x['role'] for x in row['messages']]==['user','assistant']
        print('Remote streaming passed:',task,flush=True)
    receipt=dict(repo_id=REPO,url=f'https://huggingface.co/datasets/{REPO}',revision=revision,private=info.private,
        files=len(paths),bytes=sum((ROOT/p).stat().st_size for p in paths),pairs=v['pairs'],rows_per_configuration=v['rows_per_task'],
        configurations=['acceptability','correction'],manifest_sha256=digest(ROOT/'metadata/manifest.json'),
        remote_file_set_and_hashes_verified=True,remote_streaming_load_verified=True,completed_at=datetime.now(timezone.utc).isoformat())
    (EVIDENCE/'dutch-hf-upload.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
