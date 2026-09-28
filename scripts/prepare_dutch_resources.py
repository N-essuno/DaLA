"""Download the pinned Dutch lexical resource and verify its checksum."""
from pathlib import Path
import hashlib
import requests
from dala.profiles import load_profile, resource


def main():
    profile=load_profile('nl');path=resource(profile,'lexicon');path.parent.mkdir(parents=True,exist_ok=True)
    root=profile['lexicon_source']
    for filename in ['wordlist.txt','LICENSE.txt']:
        response=requests.get(root+'/'+filename,timeout=60);response.raise_for_status()
        if filename=='wordlist.txt' and hashlib.sha256(response.content).hexdigest()!=profile['lexicon_sha256']:
            raise ValueError('Downloaded Dutch word list checksum mismatch')
        (path.parent/filename).write_bytes(response.content)
    print(path)


if __name__=='__main__':main()
