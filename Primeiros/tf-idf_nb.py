import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string

import nltk


train = pd.read_csv('train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']

classes = ["TRUTHFULPOSITIVE", "TRUTHFULNEGATIVE", "DECEPTIVENEGATIVE", "DECEPTIVEPOSITIVE"]

##################################################################################
# filtrar nomes usando spacy
import spacy
nlp = spacy.load('en_core_web_sm')

def filter_names(text):
    doc = nlp(text)
    for X in doc.ents:
        if X.label_ == 'PERSON':
            text = text.replace(X.text, '')
    return text

train['Text'] = train['Text'].apply(filter_names)

open("filtered_names.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

##################################################################################
# lowercase
train['Text'] = train['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

open("lowercase.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

##################################################################################
# transforming <word>n't in <word> not and transform <word>'s in <word> s

def not_transform(text):
    if "n't" in text:
        return text.replace("n't", " not")
    elif "'" in text:
        return text.replace("'", " ")
    return text

train['Text'] = train['Text'].apply(not_transform)

open("apost_transformed.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

##################################################################################
# removing punctuation and numbers
def preprocess_text(text):
    # Remove numerical data
    words = text.split()
    filtered_words = [word for word in words if not word.isdigit()]
    text = ' '.join(filtered_words)
    
    # Remove non-alphanumeric characters and keep only alphabets
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])

    return text

train['Text'] = train['Text'].apply(preprocess_text)

open("no_punctuation.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

##################################################################################
# removing stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
including = ['no', 'nor', 'not', 'but', 'against', 'most', 'more']

train['Text'] = train['Text'].apply(lambda x: " ".join(x for x in x.split() if (x not in stop) or (x in including)))

open("no_stopwords.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

##################################################################################
# lemmatizing Nao funciona bem
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatizing(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

train['Text'] = train['Text'].apply(lemmatizing)

open("lemmatized.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

##################################################################################

X = train['Text']
y = train['Class']

sum = 0

for i in range(0, 1):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=i)

    pd.DataFrame(X_test).to_csv("X_test.txt", sep="\t", index=False, header=False)

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
    pd.DataFrame(predictions).to_csv("predictions.txt", sep="\t", index=False, header=False)
    pd.DataFrame(y_test).to_csv("y_test.txt", sep="\t", index=False, header=False)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    #print("Accuracy:", accuracy)

    sum = accuracy + sum

print("Average: ", sum/1)


    #nothing: Accuracy: 0.7821428571428571
    #lowercase: Accuracy: 0.7821428571428571
    #no punctuation: Accuracy: 0.7892857142857143
    # idf = True: Accuracy: 0.7892857142857143
    # no stop words: Accuracy: 0.7964285714285714
    # stemming: Accuracy: 0.7857142857142857
    # lemmatization: Accuracy: 0.7857142857142857


    # nothing x100: Accuracy: 0.754142857142857
    # lowercase x100: Accuracy: 0.754142857142857
    # no stop words x100: Accuracy: 0.7771428571428574
    # idf = True x100: Accuracy: 0.7771428571428574
    # no punctuation x100: Accuracy: 0.7718571428571427
    # lemmatization x100: Accuracy: 0.7740714285714289
    # everything x100: Accuracy: 0.7740714285714289