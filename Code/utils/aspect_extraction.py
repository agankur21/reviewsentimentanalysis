import re
from collections import Counter

import nltk
from nltk.corpus import stopwords


def aspects_from_tagged_sents(tagged_sentences):
    """
    INPUT: list of lists of strings
    OUTPUT: list of aspects
    Given a list of tokenized and pos_tagged sentences from reviews
    about a given restaurant, return the most common aspects
    """

    STOPWORDS = set(stopwords.words('english'))

    # find the most common nouns in the sentences
    noun_counter = Counter()

    for sent in tagged_sentences:
        for word, pos in sent:
            if pos == 'NNP' or pos == 'NN' and word not in STOPWORDS:
                noun_counter[word] += 1

    # list of tuples of form (noun, count)
    out = [noun for noun, _ in noun_counter.most_common(10)]
    return out


class SentenceAspectExtractor():
    # Grammar for NP chunking
    GRAMMAR = r"""
    NBAR:
        {<DET>?<NN.*|JJ|VBN>*<NN.*>*}  # Nouns and Adjectives, terminated with Nouns

    NP:
        {<NBAR><IN|CC><NBAR>}  # Above, connected with in/of/etc...
        {<NBAR>}
    """

    CHUNKER = nltk.RegexpParser(GRAMMAR)

    _my_stopword_additions = ["it's", "i'm", "star", "", "time", "night", "try", "sure", "times", "way", "friends"]
    STOPWORDS = set(stopwords.words('english') + _my_stopword_additions)

    PUNCT_RE = re.compile("^[\".:;!?')(/]$")

    # FORBIDDEN = {'great', 'good', 'time', 'friend', 'way', 'friends'}
    FORBIDDEN = {}

    def __init__(self):
        pass

    def get_sent_aspects(self, tagged_sent):
        """
        INPUT: Sentence
        OUTPUT: list of lists of strings
        Given a sentence, return the aspects
        """
        tree = SentenceAspectExtractor.CHUNKER.parse(tagged_sent)
        aspects = self.get_NPs(tree)
        # filter invalid aspects
        flat_list_aspects=[]
        for asp in aspects:
            if self.valid_aspect(asp):
                flat_list_aspects = flat_list_aspects + asp
        return flat_list_aspects

    def get_NPs(self, tree):
        """
        Given a chunk tree, return the noun phrases
        """
        return [[w for w, t in leaf] for leaf in self.leaves(tree)]

    def leaves(self, tree):
        """
        Generator of NP (nounphrase) leaf nodes of a chunk tree.
        """
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            leaves = filter(lambda t: t[1] == 'NN' or t[1] == 'NNP' or t[1] == 'NNS', subtree.leaves())
            yield leaves

    def valid_aspect(self, aspect):
        """
        INPUT: list of strings
        OUTPUT: boolean
        """

        no_stops = [w for w in aspect if w not in SentenceAspectExtractor.STOPWORDS and not self.PUNCT_RE.match(w)]

        if len(no_stops) < 1:  #
            return False
        elif any([forbid_wrd in aspect for forbid_wrd in SentenceAspectExtractor.FORBIDDEN]):
            return False
        else:
            return True
