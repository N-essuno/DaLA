"""Compile language-specific resources into shared-pipeline inputs.

Generated rules remain audit candidates. Counts are not a precision claim.
"""
from collections import defaultdict
import hashlib
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'wiki/artifacts/six-language-expansion'
RES = ROOT / 'la_output/resources/multilingual'

LANG = {
 'pl': dict(name='Polish', accept='Czy to zdanie jest poprawne gramatycznie i ortograficznie w języku polskim? Odpowiedz wyłącznie yes albo no.',
  correct='Popraw błędy gramatyczne i ortograficzne w tym polskim zdaniu. Jeśli zdanie jest poprawne, zwróć je bez zmian. Zachowaj język polski, znaczenie i styl. Zwróć tylko zdanie.',
  modal=['może','mogą','musi','muszą','powinien','powinna','powinni'], modal_lemmas=['móc','musieć','powinien'], perfect=[], prep={'Gen':['bez','dla','do','od'],'Dat':['dzięki']}, pronouns={'ja':'mnie','ty':'ciebie','my':'nas','wy':'was'}, subjects=['ja','ty','on','ona','ono','my','wy','oni','one'], accents={'ą':['a'],'ć':['c'],'ę':['e'],'ł':['l'],'ń':['n'],'ó':['o','u'],'ś':['s'],'ź':['z'],'ż':['z']}),
 'sv': dict(name='Swedish', accept='Är den här meningen grammatiskt korrekt och rättstavad på svenska? Svara endast med yes eller no.',
  correct='Rätta grammatik- och stavfel i den här svenska meningen. Om meningen redan är korrekt, återge den oförändrad. Behåll svenska, betydelsen och stilen. Svara endast med meningen.',
  modal=['att','kan','kunde','ska','skall','skulle','måste','bör','borde'], perfect=['har','hade'], prep={}, pronouns={'jag':'mig','du':'dig','vi':'oss'}, subjects=[], accents={'å':['a'],'ä':['a'],'ö':['o']}),
 'nb': dict(name='Norwegian Bokmål', accept='Er denne setningen grammatisk korrekt og riktig stavet på norsk bokmål? Svar bare med yes eller no.',
  correct='Rett grammatikk- og stavefeil i denne setningen på norsk bokmål. Hvis setningen allerede er korrekt, gjengi den uendret. Behold bokmål, betydningen og stilen. Svar bare med setningen.',
  modal=['å','kan','kunne','skal','skulle','må','måtte','bør','burde','vil','ville'], perfect=['har','hadde'], prep={}, pronouns={'jeg':'meg','du':'deg','vi':'oss'}, subjects=[], accents={'æ':['e'],'ø':['o'],'å':['a']}, exclusions=['eg','ikkje','kva','korleis','desse','deira']),
 'nn': dict(name='Norwegian Nynorsk', accept='Er denne setninga grammatisk korrekt og rett stava på nynorsk? Svar berre med yes eller no.',
  correct='Rett grammatikk- og stavefeil i denne setninga på nynorsk. Dersom setninga alt er korrekt, gje henne att uendra. Hald på nynorsk, tydinga og stilen. Svar berre med setninga.',
  modal=['å','kan','kunne','skal','skulle','må','måtte','bør','burde','vil','ville'], perfect=['har','hadde'], prep={}, pronouns={'eg':'meg','du':'deg','vi':'oss','me':'oss'}, subjects=[], accents={'æ':['e'],'ø':['o'],'å':['a']}, exclusions=['jeg','ikke','hva','hvordan','disse']),
 'fo': dict(name='Faroese', accept='Er hesin føroyski setningurin mállæruliga rættur og rætt stavaður? Svara bara við yes ella no.',
  correct='Rætta mállæru- og stavivillur í hesum føroyska setninginum. Er setningurin longu rættur, gev hann aftur óbroyttan. Varðveit føroyskt mál, týdning og stíl. Svara bara við setninginum.',
  modal=['at','kann','kunnu','skal','skulu','má','mugu'], perfect=['hevur','hava','hevði','høvdu'], prep={'Dat':['frá','hjá'],'Gen':['til']}, pronouns={}, subjects=['eg','tú','hann','hon','vit','tit','teir','tær'], accents={'ð':['d',''],'á':['a'],'í':['i','ý'],'ó':['o'],'ú':['u'],'ý':['y','í'],'ø':['o']}),
 'is': dict(name='Icelandic', accept='Er þessi setning málfræðilega rétt og rétt stafsett á íslensku? Svaraðu aðeins með yes eða no.',
  correct='Leiðréttu málfræði- og stafsetningarvillur í þessari íslensku setningu. Ef setningin er þegar rétt skaltu skila henni óbreyttri. Varðveittu íslenskuna, merkinguna og stílinn. Skilaðu aðeins setningunni.',
  modal=['að','getur','geta','skal','skulu','má','mega'], perfect=['hefur','hafa','hafði','höfðu'], prep={'Dat':['frá','hjá'],'Gen':['til','án']}, pronouns={}, subjects=['ég','þú','hann','hún','við','þið','þeir','þær'], accents={'ð':['d'],'þ':['t'],'á':['a'],'é':['e'],'í':['i','ý'],'ó':['o'],'ú':['u'],'ý':['y','í'],'ö':['o']})
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def compile_rules(language, spec):
    morphology = json.loads((RES / language / 'morphology.json').read_text())
    groups = defaultdict(list)
    for a in morphology: groups[(a['lemma'], a['pos'])].append(a)
    definitions = []
    for pos, relation, prefix in [('ADJ','amod','adjective'),('DET','det','determiner')]:
        for feature in ['Gender','Number','Case','Definite']:
            if feature == 'Case' and language in ['sv','nb','nn']: continue
            if feature == 'Definite' and pos == 'DET': continue
            definitions.append(dict(family=f'{prefix}_{feature.lower()}', context='nominal_agreement', feature=feature, pos=[pos], relations=[relation]))
    if spec['subjects']:
        for feature in ['Number','Person']:
            definitions.append(dict(family=f'verb_{feature.lower()}_agreement',context='finite_agreement',feature=feature,pos=['VERB','AUX'],subjects=spec['subjects']))
    controllers=set(spec['modal'])
    for entry in morphology:
        if entry['lemma'] in spec.get('modal_lemmas',[]) and entry['pos'] in {'VERB','AUX'} and entry['features'].get('VerbForm') in {'Fin','Inf'}:
            controllers.update(entry['forms'])
    definitions.append(dict(family='governed_infinitive',context='governed_verb',feature='VerbForm',pos=['VERB'],from_values=['Inf'],to_values=['Fin'],controllers=sorted(controllers)))
    if spec['perfect']:
        definitions.append(dict(family='perfect_verb_form',context='governed_verb',feature='VerbForm',pos=['VERB'],from_values=['Part','Sup'],to_values=['Inf'],controllers=spec['perfect']))
    for case, prepositions in spec['prep'].items():
        definitions.append(dict(family='preposition_case',context='preposition_case',feature='Case',pos=['NOUN'],from_values=[case],controllers=prepositions,suffix=case.lower()))
    from dala.rule_compiler import compile_inflections
    rules = compile_inflections(language, morphology, definitions, dict(
        number_ignores_definite=language in {'nb','nn','sv'}, allow_unattested=language=='fo'))
    if spec['pronouns']:
        rules.append(dict(id=language+'_subject_pronoun_case',family='pronoun_case',operator='dictionary_mapping',context='subject_pronoun',
            pos=['PRON'],sources=['ud_syntax'],evidence_kind='grammar_constraint',audit_status='pending',active=True,
            mappings={k:[v] for k,v in spec['pronouns'].items()},pairs={k:[dict(replacement=v,before={},after={})] for k,v in spec['pronouns'].items()}))
    for operation in ['transpose_internal','delete_internal','delete_doubled_consonant','duplicate_consonant','substitute_character']:
        rules.append(dict(id=language+'_spelling_'+operation,family='spelling',operator='character_edit',operation=operation,
            min_length=5,max_length=24,unicode_letters=True,consonants='bcdfghjklmnpqrstvwxzðþłńśźżć',
            substitutions=spec['accents'],requires_lexical_screen=True,sources=['dictionary'],
            evidence_kind='synthetic_nonword_generator_not_observed_substitution',audit_status='pending',active=True))
    receipt = json.loads((RES/language/'receipt.json').read_text())
    return dict(schema_version=1,language=language,edit_validator='dala.language_packs.morphology:permits',
        sources={'morphology':receipt,'dictionary':{'revision':receipt['pins']['dictionary'],'url':'https://github.com/wooorm/dictionaries'},
                 'ud_syntax':{'url':f'https://universaldependencies.org/{"no" if language in ["nb","nn"] else language}/index.html'}},rules=rules)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--languages',nargs='+',choices=list(LANG),default=list(LANG))
    args=parser.parse_args()
    targets=json.loads((ROOT/'config/multilingual_targets.json').read_text())['languages']
    selected={'pl':['govpl','samorzad_gov_pl','eurlex','biblioteka_nauki'],
              'sv':['akademiliv','forskning-framsteg','cellar'],
              'nb':['government-nob'], 'nn':['government-nno','wikipedia-nno'],
              'fo':['faroese-blark-small','sosialurin-faroese-pos','wikipedia'],
              'is':['igc-adjud','igc-law','stjornartidindi','igc-journals-22-10','wikipedia']}
    coverage=json.loads((ART/'configured-coverage.json').read_text()) if (ART/'configured-coverage.json').exists() else {}
    for language,spec in LANG.items():
        if language not in args.languages: continue
        book=compile_rules(language,spec)
        rulepath=RES/language/'rules.json.gz'
        with gzip.GzipFile(filename=str(rulepath),mode='wb',mtime=0) as stream:
            stream.write(json.dumps(book,ensure_ascii=False,separators=(',',':')).encode())
        exclusions_path = ROOT/f'config/{language}_source_exclusions.json'
        if not exclusions_path.exists(): write(exclusions_path,[])
        inventory=json.loads((ART/f'{"nb" if language=="nn" else language}-inventory.json').read_text())
        paths={p['rfilename'] for p in inventory['siblings']}; sources=[]
        for name in selected[language]:
            data=f'data/{name}/data.parquet'
            if data not in paths:data=f'data/{name}/{name}.parquet'
            meta=f'data/{name}/metadata.parquet'
            card=ART/f'{"nb" if language=="nn" else language}-{name}.md'
            import yaml
            text=card.read_text(); header=yaml.safe_load(text.split('---')[1]) if text.startswith('---') else {}
            license=header.get('license_name',header.get('license','per-record'))
            if name.startswith('government-'): license='NLOD-2.0'
            source=dict(name=name,source_value=name,repo_id=inventory['id'],revision=inventory['sha'],data_file=data,
                metadata_file=meta if meta in paths else None,license=license,
                license_evidence_url=f'https://huggingface.co/datasets/{inventory["id"]}/blob/{inventory["sha"]}/data/{name}/{name}.md',
                min_document_chars=35,annotation_filters={},quality_policy='source_and_sentence_screening',language=language)
            if language=='fo' and name=='wikipedia':source['source_value']='wiki'
            if meta in paths:
                source['annotation_filters']={'content_integrity':['complete'],'content_quality':['excellent','good']}
                source['annotation_exclusions']={'content_type':['conversational']}
            sources.append(source)
        write(ROOT/f'config/{language}_sources.json',dict(schema_version=1,sources=sources))
        settings=json.loads((RES/language/'parser-settings.json').read_text())
        families=list(dict.fromkeys(r['family'] for r in book['rules']))
        profile=dict(schema_version=1,language=language,name=f'DaLA {spec["name"]} — DynaWord',mode='pairs',
            parser='stanza',parser_backend='stanza',parser_settings=settings,parser_threads=2,
            sources={'adapter':'dynaword'},sources_config=f'../{language}_sources.json',rulebook=f'../../la_output/resources/multilingual/{language}/rules.json.gz',
            exclusions=f'../{language}_source_exclusions.json',resource_directory=f'../../la_output/resources/multilingual/{language}',
            adapter='dala.language_packs.morphology:MorphologyPack',selection={'strategy':'hash_rotate_grammar_then_spelling','max_errors':1,'priority':families},
            checker={'mode':'morphology','workers':8},prompts={'acceptability':spec['accept'],'correction':spec['correct']},
            standard_exclusions=spec.get('exclusions',[]),curation={'min_words':7,'max_words':35,'max_sentence_chars':320,'max_paragraph_chars':6000},
            description='Pinned DynaWord prose. Candidate dataset: dictionary and syntactic-rule screening; independent linguistic audit required.',
            dependencies=['requests','pyarrow','huggingface-hub','stanza','spylls'],source_sampling='round_robin',
            export_used_rule_mappings=True,
            release_policy={'status':'candidate_only','require_audit_per_family':True,'target':targets[language]})
        profile['curation']['source_risk_patterns'] = {
            'is':[r'(?i)\b[A-Z]-(?:kk|kvk|hk)-(?:nf|þf|þgf|ef)\b',r'(?i)\b[A-Z]-x-x\b',r'(?i)\b(?:dags|nr|gr|mgr|sbr)\.$'],
            'nb':[r'(?i)\b(?:pkt|kap|jf|nr|meld|pst|mill)\.$'], 'nn':[r'(?i)\b(?:pkt|kap|jf|nr|meld|pst|mill)\.$']
        }.get(language, [])
        profile['curation']['source_risk_patterns'].append(r'^(?:[IVXLCDM]+|[A-Z])[.)]\s')
        profile['indefinite_articles']={'pl':[],'sv':['en','ett'],'nb':['en','ei','et'],
            'nn':['ein','ei','eit','eitt'],'fo':['ein','eina','einum','eini','eitt','einar','einari'],'is':[]}[language]
        if language=='fo':
            profile['personal_pronoun_features']={w:dict(Person=person,Number=number) for w,person,number in
                [('eg','1','Sing'),('tú','2','Sing'),('hann','3','Sing'),('hon','3','Sing'),
                 ('vit','1','Plur'),('tit','2','Plur'),('teir','3','Plur'),('tær','3','Plur')]}
        if language in {'pl','sv'}: profile['checker']['source_languagetool'] = True
        profile['morphology_options']=dict(lemma_strategy='surface_unique' if language=='fo' else 'parsed',
            source_pronoun_agreement=language in {'pl','fo','is'},complete_verb_agreement_from_subject=language=='fo',
            gender_equivalences=[['Masc','Fem']] if language=='nb' else [])
        profile_path=ROOT/f'config/languages/{language}.json'
        if profile_path.exists():
            previous=json.loads(profile_path.read_text())
            if previous.get('checker',{}).get('tools_dir'):
                profile['checker']['tools_dir']=previous['checker']['tools_dir']
        write(profile_path,profile)
        coverage[language]=[dict(id=r['id'],family=r['family'],active=r['active'],
                                mappings=sum(len(v) for v in r.get('mappings',{}).values()),audit_status=r['audit_status']) for r in book['rules']]
        print(language,'rules',len(book['rules']),'active',sum(r['active'] for r in book['rules']),flush=True)
    write(ART/'configured-coverage.json',coverage)


if __name__=='__main__':main()
