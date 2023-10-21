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

train = pd.read_csv('../../train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']

################################################################################################
# Preprocessing

stop = stopwords.words('english')
nlp = spacy.load("en_core_web_sm")
including = ['no', 'nor', 'not', 'but', 'against', 'only', 'if']
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # lowercase
    text = text.lower()
    #transforming <word>n't in <word> not
    text = text.replace("n't", " not")
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    #Remove stopwords
    text = ' '.join([word for word in text.split() if ((word not in stop) or (word in including))])
    # lemmatizing and Stemming
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()]) # stemmer.stem(word)

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

X = train['Text']
y = train['Class']

##################################################################################
# Vectorize the text data using TF-IDF

vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 3), sublinear_tf=True, max_features=20000)

################################################################################
# suport vector machine

svc = svm.SVC(kernel='linear', class_weight='balanced')

################################################################################
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

model = make_pipeline(vectorizer, svc)

cv_predictions = cross_val_predict(model, X, y, cv=5)

pd.DataFrame(cv_predictions).to_csv("modelo3_t3.txt", sep="\t", index=False, header=False)

print("Accuracy: ", accuracy_score(y, cv_predictions)) 


##################################################################################
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y, cv_predictions, labels = ['TRUTHFULPOSITIVE', 'TRUTHFULNEGATIVE', 'DECEPTIVEPOSITIVE','DECEPTIVENEGATIVE'])

labels = ['TRUTHFULPOSITIVE', 'TRUTHFULNEGATIVE', 'DECEPTIVEPOSITIVE','DECEPTIVENEGATIVE']

# Print the results
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(8, 6), dpi=100)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, annot_kws={"size": 16}, square=True)

plt.xlabel('Predicted')
plt.ylabel('Real')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.title('Confusion Matrix')
plt.show()


##################################################################################
# Accuracy: 0.8407142857142857