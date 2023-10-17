import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re
import nltk
#import string
#from nltk.corpus import stopwords
#from nltk.stem import SnowballStemmer
#from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

import spacy

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# load data
train = pd.read_csv('train.txt', sep='\t', header=None)

classes = ["TRUTHFULPOSITIVE", "TRUTHFULNEGATIVE", "DECEPTIVENEGATIVE", "DECEPTIVEPOSITIVE"]


train.columns = ['Class', 'Text']

###################################################################################
# Lematização

# nlp = spacy.load('en_core_web_sm')

# def lemmatization(text):
#      doc = nlp(text)
#      lemma_list = [token.lemma_ for token in doc]
#      return ' '.join(lemma_list)

# result = []
# for text in train['Text']:
#     result = result + [lemmatization(text)]

# train['Text'] = result


###################################################################################
# Separar teste e treino

# Assuming 'df' is your DataFrame with 'Text' and 'Class' columns
X = train['Text']
y = train['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

################################################################################

##############################################################################
# Vectorize the text data using TF-IDF

vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', use_idf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

################################################################################

# Naive Bayes
# Initialize a Multinomial Naive Bayes classifier
nb = MultinomialNB()

# Fit the classifier to the training data
nb = nb.fit(X_train_tfidf, y_train)

# Make predictions on test data
predictions = nb.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

################################################################################

################################################################################

################################################################################
# ??????? Nao fazemos a minima ideia do que seja

# Initialize the "CountVectorizer" object, which is scikit-learn's
#bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                                tokenizer = None,    \
                                preprocessor = None, \
                                stop_words = 'english',   \
                                lowercase= True,     \
                                max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(X_train)
test_data_features = vectorizer.transform(X_test)

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()
test_data_features = test_data_features.toarray()

##################################################################################
# Naive Bayes 
clf = MultinomialNB()

# Train the classifier
clf.fit(train_data_features, y_train)

##################################################################################
# Make predictions on test data
predictions = clf.predict(test_data_features)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

###################################################################################

###################################################################################

###################################################################################
# Vectorize the text data using TF-IDF

vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', use_idf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Jaccard Similarity

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

result = -1
resultado = []
for i in range(len(X_test)):
    y_result = ""
    result = -1
    for j in range(len(X_train)):
        new_result = jaccard_similarity(X_test.iloc[i], X_train.iloc[j])
        if result > new_result or result == -1:
            result = new_result
            y_result = y_train.iloc[j]
    resultado = resultado + [y_result]


print("test: ", y_test[:5])
print("predictions: ", resultado[:5])
accuracy = accuracy_score(y_test, resultado)
print("Accuracy Jaccard:", accuracy)

###################################################################################

###################################################################################

###################################################################################

# Cosine Similarity

def cosine_similarity(query, document):
    return np.dot(query, document)/(np.linalg.norm(query)*np.linalg.norm(document))

###################################################################################

###################################################################################

###################################################################################


# result = -1
# resultado = []
# for i in range(len(X_test)):
#     y_result = ""
#     result = -1
#     for j in range(len(X_train)):
#         new_result = jaccard_similarity(X_test.iloc[i], X_train.iloc[j])
#         if result > new_result or result == -1:
#             result = new_result
#             y_result = y_train.iloc[j]
#     resultado = resultado + [y_result]