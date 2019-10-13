import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#Citesc fisierul cu datele de antrenare, acesta contine 31962 linii si 3 coloane
df = pd.read_csv('train.csv', delimiter=',')

labels = df['label'].values
mesages = df['tweet'].values

#Vreau sa vad numarul de tweet-uri cu label 0 si numarul de tweet-uri cu label 1
neg_examples = np.sum(labels) / labels.shape[0]
pos_examples = 1 - neg_examples

print("\nNumarul de exemple negative: " + str(neg_examples))
print("Numarul de exemple pozitive: " + str(pos_examples))

#Prin analiza datelor se poate observa ca acestea sunt deja randomizate (d.p.d.v. al labelurilor)
#In plus, labelurile acestora sunt deja binarizate.

#Vreau sa impart fisierul cu date in 2 parti, Train si Test folosind Stratified Shuffle Split
shuffle_stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

for train_index, test_index in shuffle_stratified.split(mesages, labels):
    msg_train, msg_test = mesages[train_index], mesages[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]

print(msg_train)    
#Folosesc CountVectorizer pentru a obtine datele in format bag of words
count_vectorizer = CountVectorizer(lowercase=True, analyzer='word', stop_words='english')
count_vectorizer.fit(msg_train)
x_train = count_vectorizer.transform(msg_train)
x_test = count_vectorizer.transform(msg_test)


model = MultinomialNB(alpha=0.01)
model.fit(x_train, labels_train)

predictions = model.predict(x_test)

print("\nAcuratetea modelului este: ")
print(accuracy_score(labels_test, predictions))
print("Raportul de clasificare este: ")
print(classification_report(labels_test, predictions))
