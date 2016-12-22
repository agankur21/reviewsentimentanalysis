"""
File for aspect scoring functions
"""

from __future__ import division

import os

from tokenizer import *


class UnsupervisedLiu():
    """
    Class for scoring sentences using Bing Liu's Opinion Lexicon.
    Source:
    Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews."
       Proceedings of the ACM SIGKDD International Conference on Knowledge
       Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle,
       Washington, USA,
    Download lexicon at: http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
    """

    PATH_TO_LEXICONS = os.path.join(os.getcwd(), '../../Data')

    def __init__(self):
        """
        Read in the lexicons.
        """

        pos_path = self.PATH_TO_LEXICONS + "/positive-words.txt"
        neg_path = self.PATH_TO_LEXICONS + "/negative-words.txt"

        self.pos_lex = self.read_lexicon(pos_path)
        self.neg_lex = self.read_lexicon(neg_path)

    def read_lexicon(self, path):
        '''
        INPUT: LiuFeaturizer, string (path)
        OUTPUT: set of strings
        Takes path to Liu lexicon and
        returns set containing the full
        content of the lexicon.
        '''

        start_read = False
        lexicon = set()  # set for quick look-up
        with open(path, 'r') as f:
            for line in f:
                if start_read:
                    lexicon.add(line.strip())
                if line.strip() == "":
                    start_read = True
        return lexicon

    def predict_token_score(self, tok):
        if tok in self.pos_lex:
            return 1
        elif tok in self.neg_lex:
            return -1
        elif "_NEG" in tok:
            word_prefix = tok.split("_NEG")[0]
            if word_prefix != "":
                return -1 * self.predict_token_score(word_prefix)
        else:
            return 0

    def predict(self, tokenized_sent):
        '''
        INPUT: list of strings
        OUTPUT:
        Note: tokens should be a list of
        lower-case string tokens, possibly
        including negation markings.
        '''

        # features = {}

        doc_len = len(tokenized_sent)
        assert doc_len > 0, "Can't featurize document with no tokens."
        token_sentiment = []
        for tok in tokenized_sent:
            token_sentiment.append(self.predict_token_score(tok))
        sum_score = sum(token_sentiment)
        score = 1.0 * sum_score / doc_len
        return score



class SentimentScorer():
    """
    Class to score the sentiment of a sentence
    (from a pre-trained model).
    """

    def __init__(self):
        # unpickle the model
        # self.model = unpickle(model.pickle)
        self.model = UnsupervisedLiu()
        self.tokenizer = CustomizedTokenizer(preserve_case=True)
        self.negative_suffix_appender = NegationSuffixAdder()

    def score(self, sentence):
        """
        INPUT: SentimentScorer, string or list of strings
        OUTPUT: int in {-1, 0, +1}

        Given a sentence (tokenized or not), return a sentiment score
        for it.
        """

        # SKETCH:

        if isinstance(sentence, str):
            sentence = self.featurize(sentence)
        elif not isinstance(sentence, list):
            raise TypeError, "SentimentScorer.score got %s, expected string or list" % type(sentence)

        score = self.model.predict(sentence)
        return score

    def featurize(self, sentence):
        """
        INPUT: SentimentScorer, string
        OUTPUT:
        Given a sentence, return a featurized version
        that can be consumed by the self.model's predict method.
        """
        tokens = self.tokenizer.tokenize(sentence)
        tokens = self.negative_suffix_appender.add_negation_suffixes(tokens)
        return tokens


def demo_score_aspect():
    """
    Demo the score aspect functionality
    """

    ss = SentimentScorer()

    pos_sent = "This is good, awesome, perfect"
    neg_sent = "This is a bad, negative sentence"

    print "Score for '%s' is %f" % (pos_sent, ss.score(pos_sent))
    print "Score for '%s' is %f" % (neg_sent, ss.score(neg_sent))


if __name__ == "__main__":
    demo_score_aspect()
