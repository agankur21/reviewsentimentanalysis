# Source: src/spacy_postagging.py
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import nltk
from spacy.en import English
import aspect_extraction
import tokenizer as tknr

class POS_Tagger:
    def __init__(self):
        self.nlp = English()

    def get_pos_tags(self,content):
        doc = self.nlp(content.decode("utf-8"))
        sents = []
        for sent in doc.sents:
            sents.append(sent)
        i = 0
        tagged_sents = []
        for sent in sents:
            i += 1
            token_tags = []
            toks = self.nlp(sent.text.encode("ascii", "ignore").decode("utf-8"))
            for tok in toks:
                token_tags.append((tok.text, tok.tag_))
            clean_token_tags = []
            for tt in token_tags:
                if tt[1] == u"SPACE" or tt[1] == u"," or tt[1]== u"SP" :
                    continue
                clean_token_tags.append((tt[0], tt[1]))
            tagged_sents.append(clean_token_tags)
        return tagged_sents

def clean_tokens_tags(token_tags):
    clean_token_tags = []
    for tt in token_tags:
        if tt[1] == u"SPACE" or tt[1] == u",":
            continue
        clean_token_tags.append((tt[0], tt[1]))
    return clean_token_tags


def pos_tag_sentence(sentence,tokenizer,neg_suffix_appender=None):
    """
    INPUT: a sentence string
    OUTPUT: list of tuples
    Given a tokenized sentence, return
    a list of tuples of form (token, POS)
    where POS is the part of speech of token
    """
    tokens = tokenizer.tokenize(sentence)
    if neg_suffix_appender is not None:
        tokens = neg_suffix_appender.add_negation_suffixes(tokens)
    return clean_tokens_tags(nltk.pos_tag(tokens))



if __name__ == '__main__':
    input_text = "Carts are in excellent shape and provide you with electronic scoring and course GPS"
    aspect_extractor = aspect_extraction.SentenceAspectExtractor()
    tokenizer  = tknr.CustomizedTokenizer(preserve_case=True)
    neg_suffix_appender = tknr.NegationSuffixAdder()
    # sentences = nltk.sent_tokenize(input_text)
    # print (sentences)
    # for sentence in sentences:
    #     pos_tags = pos_tag_sentence(sentence,tokenizer,None)
    #     print(pos_tags)
    #     print (aspect_extractor.get_sent_aspects(pos_tags))
    pos_tagger= POS_Tagger()
    pos_tags_list = pos_tagger.get_pos_tags(input_text)
    for pos_tags in pos_tags_list:
        print (pos_tags)
        print(aspect_extractor.get_sent_aspects(pos_tags))

