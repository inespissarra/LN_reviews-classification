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
from nltk.tokenize import word_tokenize



nltk.download('stopwords')


train = pd.read_csv('../train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']


################################################################################################
# Preprocessing

stop = stopwords.words('english')
including = ['no', 'nor', 'not', 'but', 'against', 'only']
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    # lowercase
    text = text.lower()
    # transforming <word>n't in <word> not from words
    text = text.replace("n't", " not")
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
        else:
            # lemmatizing and Stemming from words
            words[i] = lemmatizer.lemmatize(stemmer.stem(words[i]))
            # not <word> -> NOT_word se word for adjetivo (ou NEVER)
            if words[i]=="not" and (i+1)<len(words) and nlp(words[i+1])[0].pos_=="ADJ":
                words[i] = ""
                words[i+1] = "NOT_" + words[i+1]
                i = i+1
            elif words[i]=="never" and (i+1)<len(words):
                words[i] = ""
                words[i+1] = "NEVER_" + words[i+1]
                i = i+1
        if words[i]!="":
            text = text + " " + words[i]
        i = i+1
    return text

train['Text'] = train['Text'].apply(preprocess)

open("preprocessed.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

X = train['Text']
y = train['Class']

##################################################################################
# Vectorize the text data using TF-IDF

vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 3), sublinear_tf=True, max_features=20000)

################################################################################
# suport vector machine

svc = svm.SVC(kernel='linear', class_weight='balanced')

################################################################################

model = make_pipeline(vectorizer, svc)

cv_scores = cross_val_score(model, X, y, cv=6)

print("Accuracy: ", np.mean(cv_scores))

################################################################################

# sem nots  0.8407003900566132 com if  0.8385544673098321
# tirando as palavras depois dos nots 0.8456983725224069 com if  0.8435555066456354
# com nots ;  com if  0.8435524497756258

################################################################################
