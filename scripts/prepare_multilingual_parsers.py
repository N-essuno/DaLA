"""Prepare CPU Stanza parsers and record all model-file checksums."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

PACKAGES = {'pl':'pdb','sv':'talbanken','nb':'bokmaal','nn':'nynorsk','fo':'farpahc','is':'icepahc'}


def prepare_settings(language, processors, directory):
    """Fetch and pin a configured parser; inventory choices belong to callers."""
    import stanza
    directory=Path(directory)
    downloads={k:v for k,v in processors.items() if v!='identity'}
    stanza.download(language,model_dir=str(directory),processors=downloads,package=None,verbose=False)
    hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((directory/language).rglob('*.pt'))}
    settings=dict(directory=str(directory),processors=processors,model_sha256=hashes,stanza_version=stanza.__version__)
    settings['language']=language
    return settings


def prepare(language):
    import stanza
    root=Path('la_output/resources/multilingual'); package=PACKAGES[language]
    processors={'tokenize':package,'pos':package+'_nocharlm','depparse':package+'_nocharlm',
                'lemma':'identity' if language=='fo' else package+'_nocharlm'}
    if language in {'pl','fo','is'}:processors['mwt']=package
    directory=root/'stanza'
    settings=prepare_settings(language, processors, directory)
    hashes=settings['model_sha256']
    (root/language/'parser-settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    print(language,len(hashes),'model files pinned',flush=True)


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(prepare,PACKAGES))
