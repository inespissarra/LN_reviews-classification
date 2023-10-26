import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string

import nltk


train = pd.read_csv('../train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']

##################################################################################
# lowercase
train['Text'] = train['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

##################################################################################
# removing stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

train['Text'] = train['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

##################################################################################

X = train['Text']
y = train['Class']

##################################################################################
# Vectorize the text data using TF-IDF

vectorizer = TfidfVectorizer(use_idf=True)

################################################################################
# Naive Bayes Classifier
nb = MultinomialNB()

################################################################################
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline

model = make_pipeline(vectorizer, nb)

cv_predictions = cross_val_predict(model, X, y, cv=5)

pd.DataFrame(cv_predictions).to_csv("modelo2_t2.txt", sep="\t", index=False, header=False)

print("Accuracy: ", accuracy_score(y, cv_predictions)) 

with open('predicted_vs_true.txt', 'w') as output_file:
    for review, expected, predicted in zip(X, y, cv_predictions):
        output_file.write(f"Filtered Review: {review}\nExpected Label: {expected}, Predicted Label: {predicted}\n")

##################################################################################
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y, cv_predictions, labels = ['TRUTHFULPOSITIVE', 'TRUTHFULNEGATIVE', 'DECEPTIVEPOSITIVE','DECEPTIVENEGATIVE'])

labels = ['TRUTHFULPOSITIVE', 'TRUTHFULNEGATIVE', 'DECEPTIVEPOSITIVE','DECEPTIVENEGATIVE']

# Print the results
print("Confusion Matrix TF-IDF & NB:")
print(cm)
plt.figure(figsize=(8, 6), dpi=100)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, annot_kws={"size": 16}, square=True)

plt.xlabel('Predicted')
plt.ylabel('Real')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.title('Confusion Matrix TF-IDF & NB')
plt.show()


##################################################################################
# Accuracy:  0.7878571428571428