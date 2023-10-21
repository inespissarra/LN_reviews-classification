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

train = pd.read_csv('../train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']


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
        if ((i-1) >= 0 and tokens[i-1] in negation) and \
            (((i-2)>=0 and tokens[i-2]!="if") or (i-2)<0):                                  # ver melhor, neg e pos
            pol['compound'] = 0 - pol['compound']
        if pol['compound'] > 0:
            pos = pos + pol['compound']
        else:
            neg = neg - pol['compound']
        if nltk.pos_tag([tokens[i]])[0][1] in ["JJ", "JJR", "JJS"]:                              # isto está a funcionar?  
            freq_adjectives = freq_adjectives + 1
    pos = pos / len(tokens)
    neg = neg / len(tokens)
    polarity = pos - neg
    freq_adjectives = freq_adjectives / len(tokens)
    features = [polarity, freq_adjectives] #, freq_adjectives, pos, neg,]
    return features


##################################################################################
#               Creates different vectors (features, tags and tokens)
##################################################################################
X = [review2features(review) for review in train['Text']]

y = train['Class']


##################################################################################
#                                      TF-IDF
##################################################################################

tfidf = TfidfVectorizer(use_idf=True, ngram_range=(1, 3), sublinear_tf=True, max_features=20000)
tfidf_matrix = tfidf.fit_transform(train['Text']).toarray()


##################################################################################
#                                     Combine
##################################################################################

combined_features = np.hstack((tfidf_matrix, np.array(X)))

##################################################################################
#                                     SVM
##################################################################################
clf = svm.SVC(kernel='linear', class_weight='balanced')

################################################################################

# Cross Validation
cv_predictions = cross_val_predict(clf, combined_features, y, cv=5)

pd.DataFrame(cv_predictions).to_csv("modelo4.txt", sep="\t", index=False, header=False)

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
# Accuracy:  0.8485714285714285