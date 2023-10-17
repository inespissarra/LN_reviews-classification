import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy

nltk.download('stopwords')


train = pd.read_csv('train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']

classes = ["TRUTHFULPOSITIVE", "TRUTHFULNEGATIVE", "DECEPTIVENEGATIVE", "DECEPTIVEPOSITIVE"]

################################################################################################
# Preprocessing

stop = stopwords.words('english')
including = ['no', 'nor', 'not', 'but', 'against', 'most', 'more', 'over', 'just', 'same', 'must'] # rever 
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    # lowercase
    text = text.lower()
    # transforming <word>n't in <word> not
    # if "can't" in text or "cannot" in text:
    #     text = text.replace("can't", "can not")
    #     text = text.replace("cannot", "can not")
    # if "won't" in text:
    #     text = text.replace("won't", "will not")
    if "n't" in text:
        text = text.replace("n't", " not")
    # transforming <word>'s in <word> s
    if "'" in text:
        text = text.replace("'", " ")
    # Remove non-alphanumeric characters and keep only alphabets
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    #Remove stopwords
    text = ' '.join([word for word in text.split() if (word not in stop) or (word in including)])
    # lemmatizing   
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    #not <word> -> NOT_word se word for adjetivo usando spacy (piora)
    i = 0
    words = text.split()
    while i < len(words):
        if words[i]=="not" and (i+1)<len(words) and nlp(words[i+1])[0].pos_=="ADJ": #  or nlp(words[i+1])[0].pos_=="VERB"
            text = text.replace("not " + words[i+1] + " ", "NOT_" + words[i+1] + " ")
            i = i+1
        i = i+1
        
    return text

train['Text'] = train['Text'].apply(preprocess)

open("preprocessed.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

X = train['Text']
y = train['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##################################################################################

# Jaccard Similarity

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

# create matrix
jaccard_matrix = []

for i in range(len(X_test)):
    jaccard_matrix.append([])
    for j in range(len(X_train)):
        jaccard_matrix[i].append(jaccard_similarity(X_test.iloc[i].split(), X_train.iloc[j].split()))
    
# print(jaccard_matrix)


################################################################################

# Classifier depending on jaccard

def classifier(jaccard_matrix, y_train):
    predictions = []
    for i in range(len(jaccard_matrix)):
        max = -1
        index = -1
        for j in range(len(jaccard_matrix[i])):
            if jaccard_matrix[i][j] > max:
                max = jaccard_matrix[i][j]
                index = j
        predictions.append(y_train.iloc[index])
    return predictions

predictions = classifier(jaccard_matrix, y_train)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy Jaccard:", accuracy)