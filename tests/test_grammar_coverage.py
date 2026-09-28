import unittest
from collections import defaultdict
from dala.rule_compiler import compile_inflections
from dala.language_packs.morphology import MorphologyPack


def entry(word, features, pos='ADJ', lemma='x'):
    return dict(lemma=lemma,pos=pos,forms=[word],features=features,normative=True)

def compile_rule(rows, **extra):
    rule=dict(family='agreement',context='nominal_agreement',feature='Case',pos=['ADJ'],**extra)
    return compile_inflections('xx',rows,[rule],{'preserve_all_features':True,'ignore_gender_animacy_features':[]})[0]

class GrammarCoverageTests(unittest.TestCase):
    def test_case_sets_require_opt_in_and_disjoint_values(self):
        rows=[entry('aa',{'Case':'Acc,Nom'}),entry('bb',{'Case':'Dat,Gen'}),entry('cc',{'Case':'Gen,Nom'})]
        self.assertFalse(compile_rule(rows)['active'])
        r=compile_rule(rows,set_valued_feature=True)
        self.assertEqual(r['mappings']['aa'],['bb'])
        self.assertEqual(r['mappings']['bb'],['aa'])
        self.assertNotIn('cc',r['mappings'])
        rows.append(entry('bb',{'Case':'Nom'}))
        self.assertNotIn('bb',compile_rule(rows,set_valued_feature=True)['mappings'].get('aa',[]))

    def test_conditional_gender_keeps_case_animacy_and_possessor(self):
        rule=dict(family='number',context='nominal_agreement',feature='Number',pos=['ADJ'],feature_applicability={'Gender':{'Number':['Plur']}})
        base={'Case':'Nom','Number':'Sing','Gender':'Masc','Person[psor]':'1','Animacy':'Inan'}
        plural=dict(base,Number='Plur');plural.pop('Gender')
        rows=[entry('aa',base),entry('bb',plural),entry('cc',dict(plural,Case='Acc')),entry('dd',dict(plural,Animacy='Anim')),entry('ee',dict(plural,**{'Person[psor]':'3'}))]
        r=compile_inflections('xx',rows,[rule],{'preserve_all_features':True,'ignore_gender_animacy_features':[]})[0]
        self.assertEqual(r['mappings']['aa'],['bb'])
        rows.append(entry('ff',dict(base,Gender='Fem')))
        r=compile_inflections('xx',rows,[dict(rule,feature='Gender')],{'preserve_all_features':True,'ignore_gender_animacy_features':[]})[0]
        self.assertNotIn('bb',r['mappings'].get('aa',[]))

    def pack_doc(self, words, pos, heads, deps, morphs, lemmas):
        import spacy
        from spacy.tokens import Doc
        pack=MorphologyPack.__new__(MorphologyPack);pack.profile={};pack.analyses=defaultdict(list)
        doc=Doc(spacy.blank('xx').vocab,words=words,pos=pos,heads=heads,deps=deps,morphs=morphs,lemmas=lemmas)
        return pack,doc

    def test_nominal_set_context_vetoes_ambiguous_head(self):
        p,d=self.pack_doc(['aa','noun'],['ADJ','NOUN'],[1,1],['amod','ROOT'],['Case=Acc,Nom','Case=Nom'],['x','noun'])
        rule=dict(family='case',context='nominal_agreement',feature='Case',pos=['ADJ'],relations=['amod'])
        pair=dict(lemma='x',before={'Case':'Acc,Nom'},after={'Case':'Dat,Gen'})
        p.analyses[('noun','NOUN')]=[entry('noun',{'Case':'Acc,Nom'},'NOUN')]
        self.assertTrue(p.licensed_context(d[0],rule,pair))
        p.analyses[('noun','NOUN')].append(entry('noun',{'Case':'Dat'},'NOUN'))
        self.assertFalse(p.licensed_context(d[0],rule,pair))

    def test_finnish_demonstrative_requires_attributive_context(self):
        p,d=self.pack_doc(['tämä','talo'],['PRON','NOUN'],[1,1],['det','ROOT'],['Number=Sing|PronType=Dem','Number=Sing'],['tämä','talo'])
        p.analyses[('talo','NOUN')]=[entry('talo',{'Number':'Sing'},'NOUN')]
        rule=dict(family='det_number',context='nominal_agreement',feature='Number',pos=['PRON'],relations=['det'],required_features={'PronType':['Dem']})
        pair=dict(lemma='tämä',before={'Number':'Sing','PronType':'Dem'},after={'Number':'Plur','PronType':'Dem'})
        self.assertTrue(p.licensed_context(d[0],rule,pair))
        d[0].dep_='nsubj';self.assertFalse(p.licensed_context(d[0],rule,pair))

    def test_pronoun_completion_uses_lexical_input_not_verb(self):
        p,d=self.pack_doc(['Elle','travaille','.'],['PRON','VERB','PUNCT'],[1,1,1],['nsubj','ROOT','punct'],['Number=Sing','Number=Sing|Person=3|VerbForm=Fin',''],['elle','travailler','.'])
        p.profile={'personal_pronoun_features':{'elle':{'Number':'Sing','Person':'3'}},'personal_pronoun_by_dependency':True,'curation':{'min_words':7,'max_words':35,'max_sentence_chars':320}}
        p.exclusions=set()
        self.assertEqual(p.sentence_rejection(d[:],set()),'length')
        self.assertEqual(d[0].morph.get('Person'),['3'])
        d[0].set_morph('Number=Plur|Person=3')
        self.assertEqual(p.sentence_rejection(d[:],set()),'source_pronoun_lexical_conflict')

    def test_noun_subject_rejects_collectives_ambiguity_and_person_changes(self):
        p,d=self.pack_doc(['child','works'],['NOUN','VERB'],[1,1],['nsubj','ROOT'],['Number=Sing','Number=Sing|Person=3|VerbForm=Fin'],['child','work'])
        p.analyses[('child','NOUN')]=[entry('child',{'Number':'Sing'},'NOUN')]
        rule=dict(family='verb_number',context='finite_agreement',feature='Number',pos=['VERB'],subjects=[],noun_subject={'lemmas':['child']})
        pair=dict(lemma='work',before={'Number':'Sing','Person':'3','VerbForm':'Fin'},after={'Number':'Plur','Person':'3','VerbForm':'Fin'})
        self.assertTrue(p.licensed_context(d[1],rule,pair))
        self.assertFalse(p.licensed_context(d[1],dict(rule,feature='Person'),pair))
        d[0].lemma_='group';self.assertFalse(p.licensed_context(d[1],rule,pair));d[0].lemma_='child'
        p.analyses[('child','NOUN')].append(entry('child',{'Number':'Plur'},'NOUN'))
        self.assertFalse(p.licensed_context(d[1],rule,pair))
        d[0].dep_='obj';self.assertFalse(p.licensed_context(d[1],rule,pair))

if __name__=='__main__':unittest.main()
