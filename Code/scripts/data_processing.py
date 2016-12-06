import os
import sys
sys.path.append(os.path.join(os.getcwd(),"../common"))
sys.path.append(os.path.join(os.getcwd(),"../utils"))
import constants
import pos_tagger
import aspect_extraction
from collections import Counter

def normalize_sentiment_score(raw_score):
    """
    Return the normalized scores in the range of -1,0,1 for negative , neutral and positive sentiments respectively
    :param raw_score:
    :return:
    """
    raw_score=int(raw_score)
    if raw_score >=3 and raw_score <=4:
        return 1
    elif raw_score ==1 :
        return  -1
    else :
        return 0


def parse_review_aspects_for_business(print_count=20):
    """
    Parse the review sentiment file and return a dictionary with reviews for each business aggregated
    :return:
    """
    complete_file_path= os.path.join(os.getcwd(),"../../Data",constants.REVIEW_SENTIMENT_FILE_NAME)
    business_reviews_list = []
    business_aspects_dict ={}
    business_filtered_aspects_dict={}
    prev_business_id = None
    business_id_count=0
    with open(complete_file_path, 'r') as input_file:
        for line in input_file:
            business_id,review_id,review_sentence,sentiment_score,sentiment_type = line.split("\t")
            review_sentence = review_sentence.strip("'")
            if business_id != prev_business_id:
                business_id_count += 1
                if prev_business_id is not None:
                    aspects = get_aspects_for_reviews(business_reviews_list)
                    sorted_aspects = sorted(Counter(flatten_list_of_list(aspects)).items(), key=lambda x: x[1], reverse=True)
                    filtered_aspects = map(lambda x: x[0],filter(lambda x :x[1]> 1,sorted_aspects))
                    if business_id_count <= print_count:
                        print "Aspect Frequency for Business ID : %s" %business_id
                        print sorted_aspects
                    business_aspects_dict[prev_business_id] = aspects
                    business_filtered_aspects_dict[prev_business_id] = filtered_aspects
                    business_reviews_list = []
            business_reviews_list.append(review_sentence)
            prev_business_id = business_id
    aspects = get_aspects_for_reviews(business_reviews_list)
    sorted_aspects = sorted(Counter(flatten_list_of_list(aspects)).items(), key=lambda x: x[1], reverse=True)
    filtered_aspects = map(lambda x: x[0], filter(lambda x: x[1] > 1, sorted_aspects))
    business_aspects_dict[prev_business_id] = aspects
    business_filtered_aspects_dict[prev_business_id] = filtered_aspects
    return business_aspects_dict,business_filtered_aspects_dict


def get_aspects_for_reviews(reviews):
    """
    Process each review and return a list of list of aspects for each business id
    :param business_reviews_dict:
    :return: business_aspect_dict
    """
    pos_tagger_obj = pos_tagger.POS_Tagger()
    aspect_extractor_obj = aspect_extraction.SentenceAspectExtractor()
    aspects= map(lambda review : get_aspects_for_review(review,pos_tagger_obj,aspect_extractor_obj),reviews)
    return aspects


def get_aspects_for_review(review,pos_tagger_obj,aspect_extractor_obj):
    """
    Return list of aspects for each review content (Sentence or Sentence Fragment)
    :param review:
    :param pos_tagger_obj:
    :param aspect_extractor_obj:
    :return:
    """
    list_pos_tags = pos_tagger_obj.get_pos_tags(review)
    out_aspects=[]
    for pos_tags in list_pos_tags:
        aspects = aspect_extractor_obj.get_sent_aspects(pos_tags)
        out_aspects += aspects
    return out_aspects


def flatten_list_of_list(list_of_list):
    """
    A simple utility function for flattening a list of list
    :param list_of_list:
    :return:
    """
    return [val for sublist in list_of_list for val in sublist]


def get_frequency_distribution(business_aspect_dict):
    """
    Return item count for
    :param business_aspect_dict:
    :return:
    """
    out = {k : sorted(Counter(flatten_list_of_list(v)).items(),key=lambda x: x[1], reverse=True) for k,v in business_aspect_dict.iteritems()}
    """Printing Sample Frequency Distribution for 20 business id's"""
    count=0
    for business_id,freq_distrib in out.iteritems():
        print "Aspects Count for Business Id : %s"%business_id
        print out[business_id]
        count +=1
        if count >= 20:
            break
    return out


def filter_aspects(aspects_list,filtered_aspects):
    return list(set(aspects_list).intersection(filtered_aspects))


def write_aspect_file(business_aspect_dict, business_filtered_aspects_dict):
    sentiment_file_path = os.path.join(os.getcwd(), "../../Data", constants.REVIEW_SENTIMENT_FILE_NAME)
    aspect_file_path = os.path.join(os.getcwd(), "../../Data", constants.ASPECT_FILE_NAME)
    aspect_file=open(aspect_file_path,'w')
    aspect_count=0
    prev_business_id =None
    with open(sentiment_file_path, 'r') as input_file:
        for line in input_file:
            business_id, review_id, review_sentence, sentiment_score, sentiment_type = line.split("\t")
            if business_id != prev_business_id:
                aspects_list = business_aspect_dict[business_id]
                aspect_count = 0
            filtered_aspects = filter_aspects(aspects_list[aspect_count],business_filtered_aspects_dict[business_id])
            filtered_aspects_str = ",".join(filtered_aspects).encode("utf-8")
            output_line = "\t".join([business_id, review_id, review_sentence, sentiment_score, sentiment_type,filtered_aspects_str])
            aspect_file.write(output_line + "\n")
            aspect_file.flush()
            aspect_count += 1
            prev_business_id =  business_id
    aspect_file.close()


if __name__ == '__main__':
    business_aspect_dict, business_filtered_aspects_dict = parse_review_aspects_for_business()
    write_aspect_file(business_aspect_dict, business_filtered_aspects_dict)
