from sklearn import svm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
from nltk.stem import PorterStemmer
import string
from nltk.tokenize import word_tokenize
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('vader_lexicon')


##################################################################################
#                             Read the input database
##################################################################################

train = pd.read_csv('train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']

test = pd.read_csv('test_just_reviews.txt', sep='\t', header=None)


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
test = test[0].apply(preprocess)



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
    freq_adjectives = 0
    pos = 0
    neg = 0
    for i in range(len(tokens)):
        pol = sentiment.polarity_scores(tokens[i])
        if ((i-1) >= 0 and tokens[i-1] in negation):
            pol['compound'] = 0 - pol['compound']
        if pol['compound'] > 0:
            pos = pos + pol['compound']
        else:
            neg = neg - pol['compound']
        if nltk.pos_tag([tokens[i]])[0][1] in ["JJ", "JJR", "JJS"]: 
            freq_adjectives = freq_adjectives + 1
    pos = pos / len(tokens)
    neg = neg / len(tokens)
    polarity = pos - neg
    freq_adjectives = freq_adjectives / len(tokens)
    features = [polarity, freq_adjectives]
    return features


##################################################################################
#               Creates different vectors (features, tags and tokens)
##################################################################################
X_train = [review2features(review) for review in train['Text']]
X_test = [review2features(review) for review in test]

y_train = train['Class']

y1 = []
y2 = []
for i in range(len(y_train)):
    if 'TRUTHFUL' in y_train[i]:
        y1.append('TRUTHFUL')
    else:
        y1.append('DECEPTIVE')
    if 'POSITIVE' in y_train[i]:
        y2.append('POSITIVE')
    else:
        y2.append('NEGATIVE')

##################################################################################
#                                      TF-IDF
##################################################################################

tfidf = TfidfVectorizer(use_idf=True, ngram_range=(1, 3), sublinear_tf=True, max_features=20000)
tfidf_matrix = tfidf.fit_transform(train['Text']).toarray()
tfidf_matrix_test = tfidf.transform(test).toarray()


##################################################################################
#                                     Combine
##################################################################################

combined_features = np.hstack((tfidf_matrix, np.array(X_train)))
combined_features_test = np.hstack((tfidf_matrix_test, np.array(X_test)))

##################################################################################
#                                     SVM
##################################################################################
clf1 = svm.SVC(kernel='linear', class_weight='balanced')
clf1 = clf1.fit(combined_features, y1)
clf2 = svm.SVC(kernel='linear', class_weight='balanced')
clf2 = clf2.fit(combined_features, y2)

predictions1 = clf1.predict(combined_features_test)
predictions2 = clf2.predict(combined_features_test)

predictions = []
for i in range(len(predictions1)):
    predictions.append(predictions1[i] + predictions2[i])

pd.DataFrame(predictions).to_csv("results.txt", sep="\t", index=False, header=False)