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


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##################################################################################
# Vectorize the text data using TF-IDF

vectorizer = TfidfVectorizer(use_idf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

################################################################################
# Naive Bayes Classifier
nb = MultinomialNB()

# Fit the classifier to the training data
nb = nb.fit(X_train_tfidf, y_train)

# Make predictions on test data
predictions = nb.predict(X_test_tfidf)

# escrever num txt para ver
pd.DataFrame(predictions).to_csv("modelo2_t1.txt", sep="\t", index=False, header=False)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, predictions))

##################################################################################
# # Do the matrix of confusion
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, predictions, labels = ['TRUTHFULPOSITIVE', 'TRUTHFULNEGATIVE', 'DECEPTIVEPOSITIVE','DECEPTIVENEGATIVE'])

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
# Accuracy: 0.7928571428571428