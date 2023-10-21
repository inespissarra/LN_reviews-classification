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



nltk.download('stopwords')


train = pd.read_csv('../train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']

################################################################################################
# Preprocessing

stop = stopwords.words('english')
including = ['no', 'nor', 'not', 'but', 'against', 'only', 'if'] # rever 
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    # lowercase
    text = text.lower()
    # transforming <word>n't in <word> not
    text = text.replace("n't", " not")
    # Remove links 
    text = ' '.join([word for word in text.split() if not word.startswith("http")])
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    #Remove stopwords
    text = ' '.join([word for word in text.split() if (word not in stop) or (word in including)])
    # lemmatizing and Stemming
    text = ' '.join([lemmatizer.lemmatize(stemmer.stem(word)) for word in text.split()])

    #not <word> -> NOT_word se word for adjetivo usando spacy
    i = 0
    words = text.split()
    while i < len(words):
        if words[i]=="not" and (i+1)<len(words) and nlp(words[i+1])[0].pos_=="ADJ":
            text = text.replace("not " + words[i+1] + " ", "NOT_" + words[i+1] + " ")
            i = i+1
        elif words[i]=="never" and (i+1)<len(words):
            text = text.replace("never " + words[i+1] + " ", "NEVER_" + words[i+1] + " ")
            i=i+1
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

print("Accuracy: ", np.mean(cv_scores) )


# com nots sem if 0.8435585635156452 ; com if 0.8414126407688641
# sem nots sem if 0.8435555066456354 ; com if 0.8428371421933655
