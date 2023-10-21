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


train = pd.read_csv('../train.txt', sep='\t', header=None)
train.columns = ['Class', 'Text']

################################################################################################
# Preprocessing

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    # lowercase
    text = text.lower()
    #Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop])
        
    return text

train['Text'] = train['Text'].apply(preprocess)

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

pd.DataFrame(predictions).to_csv('modelo1.txt', index=False, header=False)

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

# 0.5714285714285714