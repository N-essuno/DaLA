from types import SimpleNamespace as N
import unittest
import spacy

from dala.parsing import StanzaParser
from dala.language_packs.morphology import MorphologyPack


class ExpansionParserTests(unittest.TestCase):
    def test_unalignable_sentence_does_not_destroy_neighbor_offsets(self):
        text='Du pain. They work.'
        def word(id,text,pos,head,dep):return N(id=id,text=text,upos=pos,head=head,deprel=dep,lemma=text.lower(),feats='')
        first=N(tokens=[N(text='Du',start_char=0,end_char=2,words=[word(1,'de','ADP',2,'case'),word(2,'le','DET',3,'det')]),N(text='pain',start_char=3,end_char=7,words=[word(3,'pain','NOUN',0,'root')]),N(text='.',start_char=7,end_char=8,words=[word(4,'.','PUNCT',3,'punct')])])
        second=N(tokens=[N(text='They',start_char=9,end_char=13,words=[word(1,'They','PRON',2,'nsubj')]),N(text='work',start_char=14,end_char=18,words=[word(2,'work','VERB',0,'root')]),N(text='.',start_char=18,end_char=19,words=[word(3,'.','PUNCT',2,'punct')])])
        for sentence in [first,second]:sentence.words=[w for t in sentence.tokens for w in t.words]
        class NLP:
            processors={'tokenize':True,'mwt':True}
            def __call__(self,arg,processors):return N(sentences=[first,second]) if isinstance(arg,str) else arg
        parser=StanzaParser.__new__(StanzaParser);parser.nlp=NLP();parser.vocab=spacy.blank('xx').vocab
        parser.expand_mwt=True;parser.skip_unalignable_sentences=True;parser.max_sentence_tokens=80;parser.rejections={}
        result=parser(text)
        self.assertEqual(result.doc.text,text)
        spans=list(result.sents)
        self.assertEqual(len(spans),2)
        self.assertEqual(spans[0][0].pos_,'X')
        self.assertEqual(spans[1].text,'They work.')
        self.assertEqual(spans[1].start_char,9)
        self.assertEqual(spans[1][0].head.text,'work')
        self.assertEqual(parser.rejections['parser_unalignable_sentence'],1)

    def test_observed_spelling_still_requires_a_nonword(self):
        from spacy.tokens import Doc
        doc=Doc(spacy.blank('xx').vocab,words=['Words','work','.'],spaces=[True,False,False],heads=[1,1,1],deps=['nsubj','ROOT','punct'],pos=['NOUN','VERB','PUNCT'])
        pack=MorphologyPack.__new__(MorphologyPack);pack.profile={};pack.recognized=lambda w:w in {'words','work','works'}
        rules=[dict(id='xx_test',family='spelling',operator='observed_nonword',pos=['VERB'],mappings={'work':['works','wrok']})]
        pack.by_word={'work':rules}
        edits=pack.candidates(next(doc.sents),{'rules':rules})
        self.assertEqual([e.replacement for e in edits],['wrok'])



class ObservedSelectionTests(unittest.TestCase):
    def test_observed_preference_does_not_override_grammar(self):
        from dala.edits import Corruption
        pack=MorphologyPack.__new__(MorphologyPack)
        pack.profile={'selection':{'priority':['agreement','spelling']}}
        pack.rule_priority={'observed':-1,'random':0,'grammar':0}
        observed=Corruption('observed','spelling',0,4,'word','wrod',(0,))
        random=Corruption('random','spelling',5,9,'word','wrd',(1,))
        grammar=Corruption('grammar','agreement',10,14,'word','words',(2,))
        self.assertEqual(pack.select_edits('word word word',[random,observed],42,1),[observed])
        self.assertEqual(pack.select_edits('word word word',[random,observed,grammar],42,1),[grammar])

if __name__=='__main__':unittest.main()
