from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from nltk.stem import PorterStemmer
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize


nltk.download('stopwords')


train = pd.read_csv('train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']

test = pd.read_csv('test_just_reviews.txt', sep='\t', header=None)
test.columns = ['Text']

################################################################################################
# Preprocessing

stop = stopwords.words('english')
including = ['no', 'nor', 'not', 'but', 'against', 'only'] # rever 
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    # lowercase
    text = text.lower()
    # transforming <word>n't in <word> not from words
    # text = text.replace("n't", " not")
    #tokenize
    words = word_tokenize(text)
    i = 0
    # transforming <word>n't in <word> not from words
    while i < len(words):
        # remove punctuation from words
        words[i] = ''.join([char for char in words[i] if char not in string.punctuation])
        # remove stopwords from words
        if words[i] in stop and words[i] not in including:
            words[i] = ""
        else:
            # lemmatizing and Stemming from words
            words[i] = lemmatizer.lemmatize(stemmer.stem(words[i]))
            #not <word> -> NOT_word se word for adjetivo (ou NEVER)
            # if words[i]=="not" and (i+1)<len(words) and nlp(words[i+1])[0].pos_=="ADJ":
            #     words[i] = "NOT_" + words[i+1]
            #     words[i+1] = ""
            #     i = i+1
            # elif words[i]=="never" and (i+1)<len(words):
            #     words[i] = "NEVER_" + words[i+1]
            #     words[i+1] = ""
            #     i = i+1
        if words[i]!="":
            text = text + " " + words[i]
        i = i+1
    return text

train['Text'] = train['Text'].apply(preprocess)

test['Text'] = test['Text'].apply(preprocess)

#open("preprocessed.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

X_train= train['Text']
y_train = train['Class']

X_test = test['Text']

##################################################################################

vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 3), sublinear_tf=True, max_features=20000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

################################################################################
# suport vector machine

svc = svm.SVC(kernel='linear', class_weight='balanced')

#svc = svm.SVC()
svc.fit(X_train_tfidf, y_train)

y_pred = svc.predict(X_test_tfidf)

pd.DataFrame(y_pred).to_csv("y_pred.txt", sep="\t", index=False, header=False)

################################################################################