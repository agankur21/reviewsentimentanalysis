__author__ = "adityat"
import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import xml.etree.ElementTree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def usage():
    print("Usage:")
    print("python %s <data_dir>" % sys.argv[0])
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

def calcHistogram(ip):
    d = {}
    for i in ip:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d

def classEquilization(data,classes):
    d = {}
    if len(classes) != len(data):
        print "WRONG IP"
        return
    for i in range(len(classes)):
        if classes[i] in d:
            d[classes[i]].append(data[i])
        else:
            d[classes[i]] = [data[i]]
    dt, cls = [], []
    for i in d.keys():
        if i==1:
            d[i] += d[i] + d[i]
        elif i == 2:
            d[i] += d[i] + d[i]
        dt += d[i]
        cls += [i] * len(d[i])
    return dt, cls





if __name__ == '__main__':

    #if len(sys.argv) < 2:
    #    usage()
    #    sys.exit(1)
    classes = ['pos', 'neg']

    # Read the data
    train_data, y_train = readxml("../../Data/Restaurants_Train_v2.xml")
    test_data, y_test = readxml("../../Data/Restaurants_Test_Gold.xml")
    print calcHistogram(y_train)
    train_data, y_train = classEquilization(train_data, y_train)
    print calcHistogram(y_train)
    #print len(train_data), len(y_train)
    #exit(1)
    #print len(train_data), len(train_labels), len(test_data), len(test_labels)
    '''
    for curr_class in classes:
        dirname = os.path.join(data_dir, curr_class)
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:
                content = f.read()
                if fname.startswith('cv9'):
                    test_data.append(content)
                    test_labels.append(curr_class)
                else:
                    train_data.append(content)
                    train_labels.append(curr_class)
    # Create feature vectors
    '''
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True,
                                 ngram_range=(1, 4))
    x_train = vectorizer.fit_transform(train_data)
    x_test = vectorizer.transform(test_data)
    #test_vectors = vectorizer.transform(["This is not bad", "This is not good", "I like it", "It was good but did not work"
                                            #, "Food was good but environment was bad", "awesome food", "good movie", "good food"])
    classifier_type = 'nb'
    if classifier_type == 'logit':
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)
    elif classifier_type == 'mnb':
        clf = MultinomialNB()
        clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)
    elif classifier_type == 'rf':
        #clf = RandomForestClassifier(n_estimators=100, criterion='entropy', oob_score=True)
        clf = RandomForestClassifier(n_estimators=1000, criterion='entropy', oob_score=True,class_weight={1:10,2:10,3:0.4})
        clf = clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)


    #print "train_vectors: ", train_vectors
    #print "test_vectors: ", test_vectors
    #print "prediction: ", predicted
    #print " test_data: ",y_test
    print "ACCURACY: ", sum(predicted==y_test)*1.0/len(predicted)
    print(classification_report(y_test, predicted))
    print clf.predict(vectorizer.transform(["bad","bad worse crap shit","I am going","This is bad","This is good","Food was good but environment was bad","awesome food", "good movie", "good food"]))
    #print sum(predicted==y_test)/len(predicted)
    '''
    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(train_vectors, train_labels)
    prediction_linear = classifier_linear.predict(test_vectors)

    # Perform classification with SVM, kernel=linear
    classifier_liblinear = svm.LinearSVC()
    classifier_liblinear.fit(train_vectors, train_labels)
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    '''
    '''
    # Print results in a nice table
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))
    '''