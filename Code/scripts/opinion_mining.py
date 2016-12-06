import os
import sys

sys.path.append(os.path.join(os.getcwd(), "../utils"))
import numpy as np
import aspect_extraction as ae
import pos_tagger as pt
import scoring

pos_tagger = pt.POS_Tagger()


def get_reviews_for_business(bus_id, df):
    """
    INPUT: business id, pandas DataFrame
    OUTPUT: Series with only texts

    For a given business id, return the review_id and
    text of all reviews for that business.
    """
    return df.text[df.business_id == bus_id]


def extract_aspects(reviews):
    """
    INPUT: iterable of strings (pd Series, list)
    OUTPUT: list of aspects

    Return the aspects from the set of reviews
    """
    aspect_extractor = ae.SentenceAspectExtractor()
    tagged_tokenized_sentences = pos_tagger.get_pos_tags(reviews)
    # from the pos tagged sentences, get a list of aspects
    aspects = set([])
    for tagged_tokenized_sentence in tagged_tokenized_sentences:
        aspects.union(set(aspect_extractor.get_sent_aspects(tagged_tokenized_sentence)))
    return aspects


def get_sentences_by_aspect(aspect, reviews):
    """
    INPUT: string (aspect), iterable of strings (full reviews)
    OUTPUT: iterable of strings
    Given an aspect and a list of reviews, return a list
    sof all sentences that mention that aspect.
    """
    # tokenize each sentence
    doc = pos_tagger.nlp(reviews.decode("utf-8"))
    tokenized_sentences = [sent.string.strip() for sent in doc]
    return [sent for sent in tokenized_sentences if aspect in sent]


def score_aspect(reviews, aspect):
    """
    INPUT: iterable of reviews, iterable of aspects
    OUTPUT: score of aspect on given set of reviews

    For a set of reviews and corresponding aspects,
    return the score of the aspect on the reviews
    """
    sentiment_scorer = scoring.SentimentScorer()
    aspect_sentences = get_sentences_by_aspect(aspect, reviews)
    scores = [sentiment_scorer.score(sent) for sent in aspect_sentences]

    print scores
    return np.mean(scores)


def aspect_opinions(reviews):
    """
    INPUT: a set of reviews
    OUTPUT: dictionary with aspects as keys and values as scores
    """

    aspects = extract_aspects(reviews)
    return dict([(aspect, score_aspect(reviews, aspect)) for aspect in aspects])
