import os
import preprocessing
import constants
import plotting_utils

if __name__ == '__main__':
    data_folder_path = os.path.join(os.getcwd(),"../Data")
    sample_obs_count=100000
    #Running all the classifiers on a sample of the data to report basic accuracies
    reviews_train, y_train, reviews_test, y_test = preprocessing.load_dataset(constants.DATASET,data_folder_path,obs_count=sample_obs_count,display=True)
    #Getting tfidf features from the input training data
    features_train, features_test = preprocessing.get_tfidf_features(reviews_train=reviews_train,reviews_test=reviews_test)
    #Get Predictive accuracy of each model with the data
    accuracies = map(lambda classifier_type: preprocessing.get_raw_classifier_accuracy(classifier_type,features_train,y_train,features_test,y_test),constants.INITIAL_CLASSIFIER_LIST)
    plotting_utils.create_bar_plot(constants.INITIAL_CLASSIFIER_LIST,range(len(constants.INITIAL_CLASSIFIER_LIST)),accuracies,"Prediction Accuracy","Classifier Type","Accuracy of the Sample Data :%d on Different Classfiers"%sample_obs_count)
