import os
import sys

sys.path.append(os.path.join(os.getcwd(), "../common"))
sys.path.append(os.path.join(os.getcwd(), "../utils"))
import constants
import numpy as np
import random
from nltk.corpus import wordnet as wn
list_categories=['Food','Ambience','Value','Service','General','Irrelevant']


def get_wordnet_similar_category(word):
    """
    Get similar category using word net
    :param word:
    :return:
    """
    similarity_scores = []
    synset1 = wn.synsets(word)
    if len(synset1) == 0:
        return "Irrelevant"
    for category in list_categories:
        synset2 = wn.synsets(category.lower())[0]
        similarity_scores.append(synset1[0].path_similarity(synset2))
    return list_categories[np.argmax(similarity_scores)]


def normalize_sentiment_score(raw_score):
    """
    Return the normalized scores in the range of -1,0,1 for negative , neutral and positive sentiments respectively
    :param raw_score:
    :return:
    """
    raw_score = int(raw_score)
    if raw_score >= 3 and raw_score <= 4:
        return 1
    elif raw_score == 1:
        return -1
    else:
        return 0


def denormalize_sentiment_score(sentiment_score):
    """
    Convert the sentiment score to star rating
    :param raw_score:
    :return:
    """
    if sentiment_score >= -0.2 and sentiment_score < 0.2:
        return 3
    elif sentiment_score >= 0.2 and sentiment_score < 0.6:
        return 4
    elif sentiment_score >= 0.6:
        return 5
    elif sentiment_score <-0.2 and sentiment_score >= -0.6:
        return 2
    else:
        return 1


def get_aspect_category_map():
    in_file_path = os.path.join(os.getcwd(), "../../Data", constants.WORD_NET_SIMILARITY)
    pairs = map(lambda x: x.split(","), open(in_file_path, 'r').read().splitlines())
    return dict(filter(lambda x : x[1].strip().lower() != "irrelevant",pairs))


def write_summary(business_id,data_dictionary,out_file,num_sample_sentences=3):
    if len(data_dictionary) == 0:
        return
    for aspect_category in data_dictionary:
        average_sentiment_rating = denormalize_sentiment_score(np.mean(map(lambda x: x[0],data_dictionary[aspect_category])))
        positive_sentiment_sentences = map(lambda x : x[1],filter(lambda x : x[0] == 1,data_dictionary[aspect_category]))
        negative_sentiment_sentences = map(lambda x: x[1],
                                           filter(lambda x: x[0] == -1, data_dictionary[aspect_category]))
        num_total_sentences = len(data_dictionary[aspect_category])
        if num_total_sentences == 0:
            continue
        sample_positive_sentences = random.sample(positive_sentiment_sentences,min(num_sample_sentences * len(positive_sentiment_sentences)/num_total_sentences,num_total_sentences))
        sample_negative_sentences = random.sample(negative_sentiment_sentences,min(num_sample_sentences * len(negative_sentiment_sentences)/num_total_sentences,num_total_sentences))
        for sentence in sample_positive_sentences:
            out_file.write("%s\t%s\t%d\t(+): %s\n" %(business_id,aspect_category,average_sentiment_rating,sentence))
        for sentence in sample_negative_sentences:
            out_file.write("%s\t%s\t%d\t(-): %s\n" %(business_id,aspect_category,average_sentiment_rating,sentence))


def aggregate_aspect_rating_sample():
    aspect_category_map=get_aspect_category_map()
    input_file_path = os.path.join(os.getcwd(), "../../Data", constants.ASPECT_FILE_NAME)
    out_file_path = os.path.join(os.getcwd(), "../../Data", constants.ASPECT_BASED_SENTIMENT_SUMMARIZER)
    out_file = open(out_file_path,'w')
    prev_business_id = None
    data_dict ={}
    with open(input_file_path,'r') as f:
        for line in f:
            line = line.strip()
            business_id, review_id, review_sentence, filtered_aspects_str, sentiment_score, sentiment_type = line.split("\t")
            if prev_business_id != business_id:
                write_summary(prev_business_id,data_dict,out_file)
                data_dict = {}
                prev_business_id = business_id
            for aspect in filtered_aspects_str.split(","):
                if aspect not in aspect_category_map:
                    continue
                aspect_category = aspect_category_map[aspect]
                if aspect_category not in data_dict:
                    data_dict[aspect_category] = []
                data_dict[aspect_category].append((normalize_sentiment_score(sentiment_score),review_sentence))
        write_summary(prev_business_id, data_dict, out_file)
    out_file.close()


def get_aspect_category_match():
    """
    Get an aspect category for aspect.
    :return:
    """
    in_file_path = os.path.join(os.getcwd(), "../../Data", constants.ASPECT_CATEGORY_FILENAME)
    out_file_path = os.path.join(os.getcwd(), "../../Data", "word_net_similarity.csv")
    aspects = map(lambda x: x.strip(), open(in_file_path, 'r').readlines())
    categories = map(lambda x :get_wordnet_similar_category(x),aspects)
    out_file = open(out_file_path,'w')
    for i in range(len(aspects)):
        out_file.write(aspects[i].strip() + "," + categories[i] + "\n")
    out_file.close()

if __name__ == '__main__':
    get_aspect_category_match()
    aggregate_aspect_rating_sample()
    pass

