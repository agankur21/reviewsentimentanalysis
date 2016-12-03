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

    def get_pos_tags(self, content):
        doc = self.nlp(content.decode("utf-8"))
        sents = []
        for sent in doc.sents:
            sents.append(sent)
        i = 0
        tagged_sents = []
        for sent in sents:
            i += 1
            print("Processing sentence# %d: %s" % (i, sent))
            token_tags = []
            toks = self.nlp(sent.text.encode("ascii", "ignore").decode("utf-8"))
            for tok in toks:
                token_tags.append((tok.text, tok.tag_))
            clean_token_tags = []
            for tt in token_tags:
                if tt[1] == u"SPACE" or tt[1] == u"," :
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
    input_text = "Mr Hoagie is an institution. Walking in, it does not seem like a throwback to 30 years ago, old fashioned menu board, booths out of the 70s, and a large selection of food. Their speciality is the Italian Hoagie, and it is voted the best in the area year after year. I usually order the burger, while the patties are obviously cooked from frozen, all of the other ingredients are very fresh. Overall, its a good alternative to Subway, which is down the road.It had great fish tacos"
    aspect_extractor = aspect_extraction.SentenceAspectExtractor()
    tokenizer  = tknr.CustomizedTokenizer(preserve_case=True)
    neg_suffix_appender = tknr.NegationSuffixAdder()
    sentences = nltk.sent_tokenize(input_text)
    for sentence in sentences:
        pos_tags = pos_tag_sentence(sentence,tokenizer,None)
        print (aspect_extractor.get_sent_aspects(pos_tags))

