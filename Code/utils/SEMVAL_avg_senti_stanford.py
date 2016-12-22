#IMPORTS
import json
#import numpy as np
import sys
from pycorenlp import StanfordCoreNLP
from sklearn.metrics import classification_report
import xml.etree.ElementTree

## GLOBAL VARs
NUMCOUNT = 100000

def readxml(path):
    f_xml = path
    e = xml.etree.ElementTree.parse(f_xml)
    count = 1
    review = []
    overallPolarity = []
    for atype in e._root._children:
        #print "21##", atype.tag, atype.attrib
        ovp = 0
        text = ""
        aspect = []
        polarity = []
        for child in atype._children:
            if child.tag == 'text':
                text = child.text
                review.append(text)
            if child.tag == 'aspectCategories':
                for ch in child._children:
                    if ch.tag == 'aspectCategory':
                        aspect.append(ch.attrib['category'])
                        polarity.append(ch.attrib['polarity'])
                        polarity_temp = 1 if ch.attrib['polarity'] == 'positive' else -1 if ch.attrib['polarity'] == 'negative' else 0
                        ovp += polarity_temp
        #print count, ". text: ",text, " aspect: ", aspect, " polarity: ", polarity, " ovp: ",ovp
        overallPolarity.append(3 if ovp > 0 else 1 if ovp < 0 else 2)
        count += 1
    return review, overallPolarity

if __name__ == '__main__':

    #if len(sys.argv) < 2:
    #    usage()
    #    sys.exit(1)
    classes = ['pos', 'neg']

    # Read the data
    #train_data, y_train = readxml("../../Data/Restaurants_Train_v2.xml")

    test_data, y_test = readxml("../../Data/Restaurants_Test_Gold.xml")
    gold = []
    predicted = []
    sen = []
    count = 0
    for line in test_data:
        count += 1
        nlp = StanfordCoreNLP('http://localhost:9000')
        #print line[1].decode("utf8")
        #text = line[1]#.encode('utf-8')
        #print line
        res = nlp.annotate(line,properties={'annotators': 'sentiment','outputFormat': 'json'})
        tmp = 0
        for s in res["sentences"]:
            #temp = "%s\t%s\t%s\t%s" % (" ".join([t["word"] for t in s["tokens"]]),
            #   s["sentimentValue"], s["sentiment"], line[0])
            tmp += 1 if s["sentiment"]=="Positive" else -1 if s["sentiment"]=="Negative" else 0
        predicted.append(3 if tmp > 0 else 1 if tmp < 0 else 2)
        #if count == 100:
        #    break
    print len(y_test)
    print len(predicted)
    print(classification_report(y_test, predicted))