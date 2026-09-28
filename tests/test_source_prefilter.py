import unittest
import spacy
import stanza
from collections import Counter
from dala.parsing import StanzaParser
from dala.source_screen import source_text_rejection

class PrefilterTests(unittest.TestCase):
    def test_shared_text_gate(self):
        config={'max_sentence_chars':320,'source_risk_patterns':[r'\(\s*%\s*\)']}
        self.assertEqual(source_text_rejection('Short.',config),'length')
        self.assertEqual(source_text_rejection('The extracted percentage is missing here (%).',config),'source_extraction_risk')
        self.assertIsNone(source_text_rejection('The children played in the beautiful garden.',config))

    def test_rejected_sentence_skips_syntax_without_losing_neighbor_offsets(self):
        text='Short. The children played in the beautiful garden.'
        specifications=[['Short','.'],['The','children','played','in','the','beautiful','garden','.']]
        entries=[];cursor=0
        for words in specifications:
            row=[]
            for i,word in enumerate(words):
                start=text.index(word,cursor);cursor=start+len(word)
                row.append(dict(id=i+1,text=word,start_char=start,end_char=cursor,
                                upos='NOUN',lemma=word.lower(),head=0 if i==0 else 1,deprel='root' if i==0 else 'dep'))
            entries.append(row)
        document=stanza.Document(entries,text=text)
        class NLP:
            processors={'tokenize':True}
            def __init__(self):self.syntax_sentences=[]
            def __call__(self,arg,processors):
                if isinstance(arg,str):return document
                self.syntax_sentences.extend(arg.sentences);return arg
        parser=StanzaParser.__new__(StanzaParser)
        parser.nlp=NLP();parser.vocab=spacy.blank('xx').vocab
        parser.expand_mwt=False;parser.skip_unalignable_sentences=True
        parser.recover_mwt=True;parser.prefilter_text=True;parser.curation={'max_sentence_chars':320}
        parser.max_sentence_tokens=80;parser.rejections={};parser.stage_seconds=Counter()
        result=parser(text)
        self.assertEqual(len(parser.nlp.syntax_sentences),1)
        self.assertEqual(result.doc.text,text)
        first,second=result.sents
        self.assertEqual(first.pre_rejection,'length')
        self.assertEqual(second.start_char,7)
        self.assertEqual(second.text,text[7:])
        self.assertEqual(parser.rejections['parser_prefilter_length'],1)
        self.assertTrue(all(text[t.idx:t.end]==t.text for t in second))
