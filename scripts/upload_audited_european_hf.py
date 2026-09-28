#!/usr/bin/env python3
"""Publish only pinned validated package directories, then verify remote hashes."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
import hf_package_runtime as runtime

ROOT=Path('/work/mimir/DaLA/export-upload/european-audited-20260928')
RECEIPT=ROOT/'upload-receipts.json'
MUTEX=threading.Lock()
receipts={}

def save(repo,value):
    with MUTEX:
        receipts[repo]=value
        p=RECEIPT.with_suffix('.tmp');p.write_text(json.dumps(receipts,indent=2)+'\n');p.replace(RECEIPT)

def git_hash(path):
    h=hashlib.sha1();h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(api,repo,revision,folder,files):
    remote={f.path:f for f in api.list_repo_tree(repo,repo_type='dataset',revision=revision,recursive=True) if hasattr(f,'blob_id')}
    if set(remote)!=set(files):raise ValueError('Remote file inventory differs: '+repo)
    for name in files:
        entry=remote[name];path=folder/name
        if entry.size!=path.stat().st_size:raise ValueError('Remote size mismatch: '+name)
        if entry.lfs:
            if entry.lfs.sha256!=runtime.digest(path):raise ValueError('Remote LFS checksum mismatch: '+name)
        elif entry.blob_id!=git_hash(path):raise ValueError('Remote git blob mismatch: '+name)
    manifest=hf_hub_download(repo,'metadata/manifest.json',repo_type='dataset',revision=revision)
    if runtime.digest(manifest)!=runtime.digest(folder/'metadata/manifest.json'):raise ValueError('Remote manifest mismatch')

def upload(item):
    api=HfApi();folder=Path(item['path']);repo=item['repo_id'];assert repo.startswith('schneiderkamplab/dala-') and repo.endswith('-audited')
    manifest=json.loads((folder/'metadata/manifest.json').read_text());proof=json.loads((folder/'metadata/validation.json').read_text())
    assert proof['status']=='passed' and proof['pairs']==item['pairs']
    assert manifest['quality_status']=='automated_pair_audit_pass_only' and manifest['repo_id']==repo
    files=sorted(set(manifest['files'])|{'metadata/manifest.json','metadata/validation.json','.gitattributes'})
    assert {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}==set(files)
    for name in files:
        path=folder/name
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):raise ValueError('Unsafe upload path')
        if name in manifest['files']:
            entry=manifest['files'][name]
            if path.stat().st_size!=entry['bytes'] or runtime.digest(path)!=entry['sha256']:raise ValueError('Local package changed: '+name)
    checksum=runtime.digest(folder/'metadata/manifest.json');saved=receipts.get(repo)
    if saved and saved['manifest_sha256']!=checksum:raise ValueError('Prior upload content differs')
    try:info=api.dataset_info(repo)
    except RepositoryNotFoundError:info=None
    if info and not saved:
        # Do not overwrite a repository outside this uploader's recorded ownership.
        remote=hf_hub_download(repo,'metadata/manifest.json',repo_type='dataset',revision=info.sha)
        if runtime.digest(remote)!=checksum:raise ValueError('Existing unowned repository differs: '+repo)
        verify(api,repo,info.sha,folder,files)
        result=dict(status='verified',manifest_sha256=checksum,revision=info.sha,pairs=item['pairs'],file_count=len(files),completed=time.time(),existing_matching_repository=True)
        save(repo,result);return result
    if saved and saved['status']=='verified':
        verify(api,repo,saved['revision'],folder,files);print('VERIFIED_EXISTING',repo,flush=True);return saved
    result=dict(status='publishing',manifest_sha256=checksum,pairs=item['pairs'],authorized='User: upload them',public=True,started=time.time())
    save(repo,result)
    if not info:api.create_repo(repo,repo_type='dataset',private=False)
    commit=api.create_commit(repo,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(folder/name)) for name in files],commit_message='Publish validated audit-accepted DaLA pairs with both tasks and original splits')
    result.update(status='uploaded_verifying',revision=commit.oid,commit_url=commit.commit_url);save(repo,result)
    verify(api,repo,commit.oid,folder,files)
    result.update(status='verified',file_count=len(files),completed=time.time(),verification='Exact file set, sizes, Git/LFS hashes, downloaded manifest SHA256')
    save(repo,result);print('VERIFIED',repo,item['pairs'],commit.oid,flush=True);return result

if __name__=='__main__':
    lock=(ROOT/'.upload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if RECEIPT.exists():receipts=json.loads(RECEIPT.read_text())
    HfApi().whoami()
    inventory=json.loads((ROOT/'packages.json').read_text())
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(upload,inventory['packages']))
    print('ALL_VERIFIED',len(results),sum(r['pairs'] for r in results),flush=True)
