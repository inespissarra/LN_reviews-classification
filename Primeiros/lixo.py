# stemming Naofunciona bem
# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()

# def stemming(text):
#     return ' '.join([stemmer.stem(word) for word in text.split()])

# train['Text'] = train['Text'].apply(stemming)

# open("stemmed.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

# ##################################################################################
# removing punctuation
# def remove_punctuation(text):
#     return text.translate(str.maketrans('', '', string.punctuation))

# train['Text'] = train['Text'].apply(remove_punctuation)

# open("no_punctuation.txt", "w").write(train.to_csv(sep="\t", index=False, header=False))

# ##################################################################################

# filtrar nomes usando spacy
# import spacy
# nlp = spacy.load('en_core_web_sm')

# def filter_names(text):
#     doc = nlp(text)
#     for X in doc.ents:
#         if X.label_ == 'PERSON':
#             text = text.replace(X.text, '')
#     return text

# train['Text'] = train['Text'].apply(filter_names)

# open("filtered_names.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))






##################################################################################
# lowercase
# train['Text'] = train['Text'].apply(str.lower)

# open("lowercase.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

##################################################################################
# transforming <word>n't in <word> not and transform <word>'s in <word> s

# def not_transform(text):
#     if "n't" in text:
#         text = text.replace("n't", " not")
#     if "'" in text:
#         text = text.replace("'", " ")
#     return text

# train['Text'] = train['Text'].apply(not_transform)

# open("apost_transformed.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))


##################################################################################
# removing punctuation and numbers
# def preprocess_text(text):
#     # Remove numerical data
#     words = text.split()
#     filtered_words = [word for word in words if not word.isdigit()]
#     text = ' '.join(filtered_words)
    
#     # Remove non-alphanumeric characters and keep only alphabets
#     text = ''.join([char for char in text if char.isalpha() or char.isspace()])

#     return text

# train['Text'] = train['Text'].apply(preprocess_text)

# open("no_punctuation.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

##################################################################################
# removing stopwords
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# stop = stopwords.words('english')
# including = ['no', 'nor', 'not', 'but', 'against', 'most', 'more']

# train['Text'] = train['Text'].apply(lambda x: " ".join(x for x in x.split() if (x not in stop) or (x in including)))

# open("no_stopwords.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

##################################################################################
# lemmatizing
# from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()

# def lemmatizing(text):
#     return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# train['Text'] = train['Text'].apply(lemmatizing)

# open("lemmatized.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

##################################################################################

#not <word> -> NOT_word se word for adjetivo usando spacy (piora)
    # i = 0
    # words = text.split()
    # while i < len(words):
    #     if words[i]=="not" and (i+1)<len(words) and nlp(words[i+1])[0].pos_=="ADJ":
    #         text = text.replace("not " + words[i+1] + " ", "NOT_" + words[i+1] + " ")
    #         i = i+1
    #     i = i+1



# Dicionario##################################################################################

# from sklearn import svm
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score
# import string
# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# import spacy
# from nltk.corpus import wordnet


# nltk.download('stopwords')


# train = pd.read_csv('train.txt', sep='\t', header=None)
# train.columns = ['Class', 'Text']

# classes = ["TRUTHFULPOSITIVE", "TRUTHFULNEGATIVE", "DECEPTIVENEGATIVE", "DECEPTIVEPOSITIVE"]

# ################################################################################################
# # Preprocessing
# def find_synonyms(word):
#     synonyms = []
#     for syn in wordnet.synsets(word):
#         for lemma in syn.lemmas():
#             synonyms.append(lemma.name())
#     return synonyms

# # word = "good"
# # good_synonyms = find_synonyms(word)
# # word = "like"
# # good_synonyms += find_synonyms(word)
# # #print(f"Synonyms of {word}: {good_synonyms}")

# # word = "unhappy"
# # bad_synonyms = find_synonyms(word)
# # #print(f"Synonyms of {word}: {bad_synonyms}")

# # stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]#stopwords.words('english')
# # #print(stop)

# # for i in bad_synonyms:
# #     if i in stop:
# #         print("aaaaaaa bad:", i)

# # for i in good_synonyms:
# #     if i in stop:
# #         print("aaaaaaa good:", i)
# ############################################################################
# synonyms = []
# antonyms = []

# def synonym_antonym_extractor(phrase):
#      from nltk.corpus import wordnet

#      for syn in wordnet.synsets(phrase):
#           for l in syn.lemmas():
#                synonyms.append(l.name())
#                if l.antonyms():
#                     antonyms.append(l.antonyms()[0].name())

# synonym_antonym_extractor(phrase="love")
# ##############################################################################

# stop = stopwords.words('english')
# including = ['no', 'nor', 'not', 'but', 'against', 'most', 'more', 'over', 'just', 'same', 'must'] # rever + bad_synonyms + good_synonyms
# lemmatizer = WordNetLemmatizer()
# nlp = spacy.load("en_core_web_sm")

# # Download WordNet data (if not already downloaded)
# nltk.download('wordnet')

# # Function to find synonyms of a word

# def preprocess(text):
#     # lowercase
#     text = text.lower()
#     # transforming <word>n't in <word> not
#     if "can't" in text or "cannot" in text:
#         text = text.replace("can't", "can not")
#     if "won't" in text:
#         text = text.replace("won't", "will not")
#     if "n't" in text:
#         text = text.replace("n't", " not")
#     # transforming <word>'s in <word> s
#     if "'" in text:
#         text = text.replace("'", " ")
#     # Remove numerical data
#     text = ' '.join([word for word in text.split() if not word.isdigit()])
#     # Remove non-alphanumeric characters and keep only alphabets
#     text = ''.join([char for char in text if char.isalpha() or char.isspace()])
#     # Remove stopwords
#     text = ' '.join([word for word in text.split() if (word not in stop) or (word in including)])
#     # lemmatizing   
#     text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

#     return text

# train['Text'] = train['Text'].apply(preprocess)

# open("preprocessed.txt", "w").write(train['Text'].to_csv(sep="\t", index=False, header=False))

# X = train['Text']
# y = train['Class']

# sum = 0

# for i in range(0,1):
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

#     pd.DataFrame(X_test).to_csv("X_test.txt", sep="\t", index=False, header=False)

#     ##################################################################################
#     # Vectorize the text data using TF-IDF

#     vectorizer = TfidfVectorizer(use_idf=True)
#     X_train_tfidf = vectorizer.fit_transform(X_train)
#     X_test_tfidf = vectorizer.transform(X_test)

#     ################################################################################
#     # suport vector machine

#     svc = svm.SVC()
#     svc.fit(X_train_tfidf, y_train)

#     y_pred = svc.predict(X_test_tfidf)
    
#     pd.DataFrame(y_pred).to_csv("predictions.txt", sep="\t", index=False, header=False)
#     pd.DataFrame(y_test).to_csv("y_test.txt", sep="\t", index=False, header=False)

#     sum = sum + accuracy_score(y_test, y_pred)

# print("Accuracy: ", sum/1)
