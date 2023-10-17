from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy

nltk.download('stopwords')


train = pd.read_csv('../train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']

################################################################################################
# Preprocessing

stop = stopwords.words('english')
including = ['no', 'nor', 'not', 'but', 'against', 'only'] # rever 
lemmatizer = WordNetLemmatizer()
# stemmer = nltk.stem.PorterStemmer()
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    # lowercase
    text = text.lower()
    # transforming <word>n't in <word> not
    #text = text.replace("can't", "can not")
    #text = text.replace("cannot", "can not")
    #text = text.replace("won't", "will not")
    # text = text.replace("n't", " not")
    # transforming <word>'s in <word> s
    #text = text.replace("'", " ")

    # Remove links 
    text = ' '.join([word for word in text.split() if not word.startswith("http")])
    # Remove non-alphanumeric characters and keep only alphabets
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    # Remove stopwords
    text = ' '.join([word for word in text.split() if (word not in stop) or (word in including)])
    # lemmatizing   
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()]) # stemmer.stem(word)

    # not <word> -> NOT_word se word for adjetivo usando spacy (piora)
    # i = 0
    # words = text.split()
    # while i < len(words):
    #     if words[i]=="not" and (i+1)<len(words) and nlp(words[i+1])[0].pos_=="ADJ":
    #         text = text.replace("not " + words[i+1] + " ", "NOT_" + words[i+1] + " ")
    #         i = i+1
    #     elif words[i]=="never" and (i+1)<len(words):
    #         text = text.replace("never " + words[i+1] + " ", "NEVER_" + words[i+1] + " ")
    #         i=i+1
    #     i = i+1
        
    return text

train['Text'] = train['Text'].apply(preprocess)

open("preprocessed.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

X = train['Text']
y = train['Class']

##################################################################################

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

pd.DataFrame(X_test).to_csv("X_test.txt", sep="\t", index=False, header=False)

##################################################################################
# Vectorize the text data using TF-IDF

vectorizer = TfidfVectorizer(use_idf=True, analyzer = "word", ngram_range=(1,2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

################################################################################
# suport vector machine

svc = svm.SVC()
svc.fit(X_train_tfidf, y_train)

y_pred = svc.predict(X_test_tfidf)

pd.DataFrame(y_pred).to_csv("predictions.txt", sep="\t", index=False, header=False)
pd.DataFrame(y_test).to_csv("y_test.txt", sep="\t", index=False, header=False)
print(accuracy_score(y_test, y_pred))

##################################################################################
# # Do the matrix of confusion
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

labels = ['TRUTHFULPOSITIVE', 'TRUTHFULNEGATIVE', 'DECEPTIVEPOSITIVE','DECEPTIVENEGATIVE']

# Print the results
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(8, 6), dpi=100)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix')
plt.show()