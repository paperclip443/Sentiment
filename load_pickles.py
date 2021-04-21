import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf


documents_file= open('documents.pickle', 'rb')
documents = pickle.load(documents_file)
documents_file.close()

word_features_file= open('word_features.pickle', 'rb')
word_features = pickle.load(word_features_file)
word_features_file.close()


# document represents text file
def find_features(document):
    #words = set(document) # set() removes doublicates
    words = word_tokenize(document)
    features = {}
    for w in word_features: #for words in top 3k words
        features[w] = (w in words) #boolean of true or false if word is in top 3k or not

    return features #returns {'plot': True, 'get': True....}


file= open('NB_classifier.pickle', 'rb')
NB_classifier = pickle.load(file)
file.close()

file= open('Multinomial_NB_classifier.pickle', 'rb')
Multinomial_NB_classifier = pickle.load(file)
file.close()

file= open('Bernoulli_NB_classifier.pickle', 'rb')
Bernoulli_NB_classifier = pickle.load(file)
file.close()

file= open('Logistic_Reg_Classifier.pickle', 'rb')
Logistic_Reg_Classifier = pickle.load(file)
file.close()

file= open('SGD_Classifier.pickle', 'rb')
SGD_Classifier = pickle.load(file)
file.close()

# file= open('Support_Vector_Classifier.pickle', 'rb')
# Support_Vector_Classifier = pickle.load(file)
# file.close()

file= open('Linear_SV_Classifier.pickle', 'rb')
Linear_SV_Classifier = pickle.load(file)
file.close()

# file= open('Nu_SV_Classifier.pickle', 'rb')
# Nu_SV_Classifier = pickle.load(file)
# file.close()



voted_classifier = VoteClassifier(NB_classifier,
                                  Multinomial_NB_classifier,
                                  Bernoulli_NB_classifier,
                                  Logistic_Reg_Classifier,
                                  SGD_Classifier,
                                  Linear_SV_Classifier)




def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)










