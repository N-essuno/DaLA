"""Expand UD lemma vocabulary with SGJP and modern Icelandic BÍN paradigms.

UD supplies the vocabulary only. All generated forms come from the independent
lexicons; archaic/flagged Icelandic entries and Polish stylistically marked forms
are omitted. UD feature conversion is conservative and tested separately.
"""
from collections import defaultdict
from importlib.metadata import version
import hashlib
import itertools
import json
from pathlib import Path
import re

ROOT=Path('la_output/resources/multilingual')


def polish_features(tag):
    parts=tag.split(':'); cls=parts[0]
    if cls not in {'subst','adj','fin','inf','praet','ppron12','ppron3'}: return []
    pos={'subst':'NOUN','adj':'ADJ','fin':'VERB','inf':'VERB','praet':'VERB','ppron12':'PRON','ppron3':'PRON'}[cls]
    mapping={'sg':('Number','Sing'),'pl':('Number','Plur'),'nom':('Case','Nom'),'gen':('Case','Gen'),
             'dat':('Case','Dat'),'acc':('Case','Acc'),'inst':('Case','Ins'),'loc':('Case','Loc'),'voc':('Case','Voc'),
             'm1':('Gender','Masc'),'m2':('Gender','Masc'),'m3':('Gender','Masc'),'f':('Gender','Fem'),'n':('Gender','Neut'),
             'pos':('Degree','Pos'),'com':('Degree','Cmp'),'sup':('Degree','Sup'),'pri':('Person','1'),'sec':('Person','2'),'ter':('Person','3')}
    result=[]
    for values in itertools.product(*(p.split('.') for p in parts[1:])):
        f=dict(mapping[t] for t in values if t in mapping)
        if cls=='inf': f.update(VerbForm='Inf')
        elif cls=='fin': f.update(VerbForm='Fin',Mood='Ind')
        elif cls=='praet': f.update(VerbForm='Fin',Tense='Past',Mood='Ind')
        for t in values:
            if t in {'m1','m2','m3'}:f['Animacy']={'m1':'Hum','m2':'Anim','m3':'Inan'}[t]
        result.append((pos,f))
    return result


def icelandic_features(category,mark):
    pos={'kk':'NOUN','kvk':'NOUN','hk':'NOUN','lo':'ADJ','so':'VERB','fn':'DET','pfn':'PRON','gr':'DET'}.get(category)
    if not pos or 'OP-' in mark or 'OBEYGJANLEGT' in mark:return None
    f={}
    if category in {'kk','kvk','hk'}:f['Gender']={'kk':'Masc','kvk':'Fem','hk':'Neut'}[category]
    case=re.search(r'(ÞGF|ÞF|NF|EF)(ET|FT)',mark)
    if case:f.update(Case={'ÞGF':'Dat','ÞF':'Acc','NF':'Nom','EF':'Gen'}[case[1]],Number='Sing' if case[2]=='ET' else 'Plur')
    for code,value in [('KVK','Fem'),('KK','Masc'),('HK','Neut')]:
        if re.search(r'(?:^|-)'+code+r'(?:-|$)',mark):f['Gender']=value
    if pos=='NOUN':f['Definite']='Def' if 'gr' in mark else 'Ind'
    if pos=='ADJ':
        if mark.startswith(('FSB','FVB')):f['Degree']='Pos'
        elif mark.startswith(('ESB','EVB')):f['Degree']='Sup'
        elif mark.startswith('MST'):f['Degree']='Cmp'
        if 'SB' in mark:f['Definite']='Ind'
        elif 'VB' in mark:f['Definite']='Def'
    if pos=='VERB':
        if 'LHÞT' in mark:f.update(VerbForm='Part',Tense='Past')
        elif 'SAGNB' in mark:f['VerbForm']='Sup'
        elif 'NH' in mark:f['VerbForm']='Inf'
        elif 'FH' in mark or 'VH' in mark:
            f.update(VerbForm='Fin',Mood='Ind' if 'FH' in mark else 'Sub',Tense='Past' if 'ÞT' in mark else 'Pres')
            person=re.search(r'([123])P-(ET|FT)',mark)
            if person:f.update(Person=person[1],Number='Sing' if person[2]=='ET' else 'Plur')
        else:return None
        f['Voice']='Mid' if 'MM' in mark else 'Act'
    return pos,f


def prepare(language):
    root=ROOT/language;path=root/'morphology.json';ud=root/'ud-morphology.json'
    if not ud.exists():ud.write_bytes(path.read_bytes())
    vocabulary=json.loads(ud.read_text())
    lemmas=sorted({a['lemma'] for a in vocabulary})
    lemma_pos=defaultdict(set)
    for a in vocabulary:lemma_pos[a['lemma']].add(a['pos'])
    forms=defaultdict(set);words=set()
    if language=='pl':
        import morfeusz2
        lexicon=morfeusz2.Morfeusz()
        for lemma in lemmas:
            for word,native_lemma,tag,names,labels in lexicon.generate(lemma):
                if labels or not word.isalpha() or not word.islower() or 'nazwa_własna' in names:continue
                words.add(word)
                for pos,features in polish_features(tag):
                    forms[(native_lemma.split(':')[0],pos,tuple(sorted(features.items())))].add(word)
                    if pos in {'ADJ','PRON'} and 'DET' in lemma_pos[native_lemma.split(':')[0]]:
                        forms[(native_lemma.split(':')[0],'DET',tuple(sorted(features.items())))].add(word)
        source=dict(package='morfeusz2',version=version('morfeusz2'),url='https://morfeusz.sgjp.pl/en',dictionary='SGJP bundled with the pinned wheel')
    else:
        from islenska import Bin
        lexicon=Bin();seen=set()
        for lemma in lemmas:
            for item in lexicon.lookup(lemma)[1]:
                if item.ord!=lemma or item.bin_id in seen:continue
                seen.add(item.bin_id)
                for entry in lexicon.lookup_id(item.bin_id):
                    if (entry.einkunn!=1 or entry.beinkunn!=1 or entry.malsnid or entry.bmalsnid or entry.bgildi or entry.malfraedi
                            or not entry.bmynd.isalpha() or not entry.bmynd.islower()):continue
                    words.add(entry.bmynd)
                    analysis=icelandic_features(entry.ofl,entry.mark)
                    if analysis:
                        pos,features=analysis
                        forms[(entry.ord,pos,tuple(sorted(features.items())))].add(entry.bmynd)
        source=dict(package='islenska',version=version('islenska'),url='https://github.com/mideind/BinPackage',license='CC-BY-SA-4.0',
                    filters='headword/form grade 1; no register, special-usage or grammar flags')
    entries=[dict(lemma=lemma,pos=pos,features=dict(features),forms=sorted(values),normative=True)
             for (lemma,pos,features),values in sorted(forms.items())]
    path.write_text(json.dumps(entries,ensure_ascii=False,sort_keys=True)+'\n')
    wordfile=root/'normative-words.txt';wordfile.write_text('\n'.join(sorted(words))+'\n')
    receiptpath=root/'receipt.json';receipt=json.loads(receiptpath.read_text())
    receipt.update(morphology_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),morphology_analyses=len(entries),
                   morphology_source=source,normative_words_sha256=hashlib.sha256(wordfile.read_bytes()).hexdigest())
    receiptpath.write_text(json.dumps(receipt,indent=2)+'\n')
    print(language,len(entries),'analyses',len(words),'forms',flush=True)


if __name__=='__main__':
    from scripts.close_surface_analyses import close
    for language in ['pl','is']:
        prepare(language)
        close(language)
