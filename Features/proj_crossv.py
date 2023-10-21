from sklearn import svm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
from nltk.stem import PorterStemmer
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
import sklearn_crfsuite
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from sklearn.model_selection import cross_val_score

nltk.download('stopwords')
nltk.download('vader_lexicon')


##################################################################################
#                             Read the input database
##################################################################################

train = pd.read_csv('../train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']


##################################################################################
#                                Preprocessing
##################################################################################

stop = stopwords.words('english')
including = ['no', 'nor', 'not', 'but', 'against', 'only']
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    #tokenize
    words = word_tokenize(text)
    i = 0
    text = ""
    # transforming <word>n't in <word> not from words
    while i < len(words):
        # remove punctuation from words
        words[i] = ''.join([char for char in words[i] if char not in string.punctuation])
        # remove stopwords from words
        if words[i] in stop and words[i] not in including:
            words[i] = ""
        #else:
            # lemmatizing and Stemming from words
            words[i] = stemmer.stem(lemmatizer.lemmatize(words[i]))
        if text != "":
            text = text + " "
        if words[i]!="":
            text = text + words[i]
        i = i+1
    return text

train['Text'] = train['Text'].apply(preprocess)

print("after preprocessing")


##################################################################################
#         Extracts features (and convert them to sklearn-crfsuite format)
##################################################################################
negation = ["not", "no", "never", "neither", "nor", "none", "nobody", "nowhere", \
            "nothing", "hardly", "scarcely", "barely", "doesn't", "isn't", "wasn't", \
                "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't", "didn't", \
                    "aren't", "ain't", "without"]
sentiment = SentimentIntensityAnalyzer()
def review2features(review):
    tokens = nltk.word_tokenize(review)
    pos = 0
    neg = 0
    for i in range(len(tokens)):
        pol = sentiment.polarity_scores(tokens[i])
        if ((i-1) >= 0 and tokens[i-1] in negation) and \
            (((i-2)>=0 and tokens[i-2]!="if") or (i-2)<0):                                  # ver melhor, neg e pos
            pol['compound'] = 0 - pol['compound']
        if pol['compound'] > 0:
            pos = pos + pol['compound']
        else:
            neg = neg - pol['compound']
    pos = pos / len(tokens)
    neg = neg / len(tokens)
    polarity = pos - neg
    features = [polarity]
    return features


##################################################################################
#               Creates different vectors (features, tags and tokens)
##################################################################################
X = [review2features(review) for review in train['Text']]

y = train['Class']

print("after features")


##################################################################################
#                                      TF-IDF
##################################################################################

tfidf = TfidfVectorizer(use_idf=True, ngram_range=(1, 3), sublinear_tf=True, max_features=20000)
tfidf_matrix = tfidf.fit_transform(train['Text']).toarray()

print("after tfidf")


##################################################################################
#                                     Combine
##################################################################################

combined_features = np.hstack((tfidf_matrix, np.array(X)))

print("after combine")


##################################################################################
#                                     SVM
##################################################################################
clf = svm.SVC(kernel='linear')


# Cross Validation
scores = cross_val_score(clf, combined_features, y, cv=6)
print("Accuracy: ", np.mean(scores))


# 0.8507024687282199