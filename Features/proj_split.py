from sklearn import svm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
from nltk.stem import PorterStemmer
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
import sklearn_crfsuite

nltk.download('stopwords')

################################################################################################
#                                  Read the input database
################################################################################################


train = pd.read_csv('../train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']


################################################################################################
#                                      Preprocessing
################################################################################################

stop = stopwords.words('english')
including = ['no', 'nor', 'not', 'but', 'against', 'only']
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    # lowercase
    # text = text.lower()                                                                           necessário para features
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
        #else:
            # lemmatizing and Stemming from words
            # words[i] = stemmer.stem(lemmatizer.lemmatize(words[i]))                               necessário para features?
        if words[i]!="":
            text = text + " " + words[i]
        i = i+1
    return text

train['Text'] = train['Text'].apply(preprocess)


################################################################################################
#                              Retrieves reviews With their Tags
################################################################################################
class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda r: [(w, t) for w, t in zip(r['Text'].values.tolist(),
                                                           r['Class'].values.tolist())]
        self.grouped = self.data.groupby('Class').apply(agg_func)                                     # ver melhor.
        self.reviews = [r for r in self.grouped]                                                      # ver melhor. Dividir por frases?

    def get_next(self):
        try:
            r = self.grouped['Review: {}'.format(self.n_sent)]
            self.n_sent += 1
            print(r)
            return r
        except:
            return None

getter = SentenceGetter(train)
reviews = getter.reviews

################################################################################################
#                  Extracts features (and convert them to sklearn-crfsuite format)
################################################################################################
def word2features(sent, i):
    word = sent[i][0]
    features = {
     #   'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:], #sufixo 3 = last 3 characters
        'word[-2:]': word[-2:], #sufixo 2 = last 2 characters
        'word.isupper()': word.isupper(), # all chars are caps
        'word.istitle()': word.istitle(), # just the first
        'word.isdigit()': word.isdigit(), # the word is a digit
    }
    if i > 0: # ignore first word
        word1 = sent[i-1][0] # previous word
        features.update({ # features from previous word
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True # the word is beggining of the sentence                             # ver melhor. Primeira frase? 

    if i < len(sent)-1: # ignore last word
        word1 = sent[i+1][0] # next word
        features.update({ # features from next word
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True # the word is the end of the sentence                               # ver melhor. Ultima frase? 

    return features


################################################################################################
#                      Creates different vectors (features, tags and tokens)
################################################################################################

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

X = [sent2features(r) for r in reviews]

def sent2labels(sent):
    return [label for token, label in sent]                                                          # token??

y = [sent2labels(s) for s in reviews]

##################################################################################
# Vectorize the text data using TF-IDF

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=40)

# vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 3), sublinear_tf=True, max_features=20000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

##################################################################################
crf = sklearn_crfsuite.CRF(
    max_iterations=20, # results are much higher with 100
    all_possible_transitions=True
)

#crf.fit(X_train, y_train) # Needed in COLAB
try:
    crf.fit(X_train, y_train)
except AttributeError:
    pass

################################################################################

y_pred = crf.predict(X_test)

print(y_pred)
print(y_test)

# def flatten(list):
#     return [item for sublist in list for item in sublist]

# flat_y_pred = flatten(y_pred)
# flat_y_test = flatten(y_test)

# print(classification_report(flat_y_test, flat_y_pred))

################################################################################
