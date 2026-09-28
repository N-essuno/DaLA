from types import SimpleNamespace as N
import unittest
import spacy
from dala.surface_syntax import SurfaceDocument
from dala.language_packs.morphology import MorphologyPack

class SurfaceTests(unittest.TestCase):
    def make(self):
        text='Les enfants parlent du grand jardin.'
        forms=[('Les','DET',2,'det'),('enfants','NOUN',3,'nsubj'),('parlent','VERB',0,'root'),('de','ADP',7,'case'),('le','DET',7,'det'),('grand','ADJ',7,'amod'),('jardin','NOUN',3,'obl'),('.','PUNCT',3,'punct')]
        words=[N(id=i+1,text=t,upos=p,head=h,deprel=d,lemma=t.lower(),feats='') for i,(t,p,h,d) in enumerate(forms)]
        tokens=[]
        cursor=0
        for surface,ids in [('Les',[0]),('enfants',[1]),('parlent',[2]),('du',[3,4]),('grand',[5]),('jardin',[6]),('.',[7])]:
            a=text.index(surface,cursor);b=a+len(surface);cursor=b
            tokens.append(N(text=surface,start_char=a,end_char=b,words=[words[i] for i in ids]))
        return text,N(tokens=tokens,words=words)

    def test_source_spans_and_expanded_dependencies(self):
        text,s=self.make();doc=SurfaceDocument(text,[s],spacy.blank('xx').vocab);span=doc.sents[0]
        self.assertEqual(doc.doc.text,text);self.assertEqual(span.text,text)
        self.assertEqual(span[3].head.text,'jardin');self.assertEqual(span[4].head.text,'jardin')
        self.assertTrue(span[3].surface_protected);self.assertTrue(span[4].surface_protected)
        for t in span:
            if not t.surface_protected:self.assertEqual(text[t.idx:t.end],t.text)
        self.assertEqual(span[5].head.text,'jardin')

    def test_grammar_context_touching_expansion_is_blocked(self):
        text,s=self.make();span=SurfaceDocument(text,[s],spacy.blank('xx').vocab).sents[0]
        pack=MorphologyPack.__new__(MorphologyPack);pack.profile={}
        self.assertEqual(pack.context_rejection(span[5],{},{}),'protected_surface_context')

    def test_spelling_edits_only_exact_unprotected_spans(self):
        text,s=self.make();span=SurfaceDocument(text,[s],spacy.blank('xx').vocab).sents[0]
        pack=MorphologyPack.__new__(MorphologyPack);pack.profile={};pack.recognized=lambda w:w in {'les','enfants','parlent','de','le','grand','jardin'}
        rules=[dict(id='fr_'+w,family='spelling',operator='observed_nonword',pos=[p],mappings={w:[w+'x']}) for w,p in [('de','ADP'),('le','DET'),('jardin','NOUN')]]
        pack.by_word={w:[r] for w,r in zip(['de','le','jardin'],rules)}
        edits=pack.candidates(span,{'rules':rules})
        self.assertEqual([e.original for e in edits],['jardin'])
        for e in edits:self.assertEqual(text[e.start:e.end],e.original)
        from dala.pair_pipeline import prepare_sentence
        pack.sentence_rejection=lambda sentence,misspellings:None
        pack.profile={'selection':{'priority':['spelling']}}
        pack.rule_priority={}
        document={'document_id':'fixture','text':'Prefix\n'+text}
        prepared,reason,_=prepare_sentence(span,document,7,pack,{'rules':rules},
            {'language':'fr'},42,1,set(),{r['id']:r for r in rules})
        self.assertIsNone(reason)
        pair,_=prepared
        self.assertEqual(document['text'][pair['sentence_start']:pair['sentence_end']],text)
        self.assertEqual(pair['corrupted'],text.replace('jardin','jardinx'))

    def test_bad_offsets_fail_closed(self):
        text,s=self.make();s.tokens[3].start_char+=1
        with self.assertRaises(ValueError):SurfaceDocument(text,[s],spacy.blank('xx').vocab)

    def test_neighbor_sentence_offsets_and_dependencies(self):
        import copy
        text,first=self.make();second=copy.deepcopy(first)
        offset=len(text)+2
        for token in second.tokens:
            token.start_char+=offset;token.end_char+=offset
        combined=text+'\n\n'+text
        doc=SurfaceDocument(combined,[first,second],spacy.blank('xx').vocab)
        self.assertEqual(len(doc.sents),2)
        later=doc.sents[1]
        self.assertEqual(later.start_char,offset)
        self.assertEqual(combined[later.start_char:later.end_char],text)
        self.assertEqual(later[3].head.text,'jardin')
        self.assertTrue(all(t.head.owner is later for t in later))
