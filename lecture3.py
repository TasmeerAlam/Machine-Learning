import os
import glob
import numpy as np
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

#load all the spam emails
file_path = "D:/Tasmeer/Spring 2019/EEGR 565/enron1/spam"
emails, labels = [], []
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding = "ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)

#print(emails)

#load all the legitimate emails
file_path = "D:/Tasmeer/Spring 2019/EEGR 565/enron1/ham"
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding = "ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0)

#Test if a token is a word
def letters_only(astr):
    return astr.isalpha()

#Remove names from words and perform word stemming
def clean_test(docs):
    cleaned = []
    all_names = set(x.lower() for x in names.words()) # for all the name in the corpus
    lemmatizer = WordNetLemmatizer()
    for doc in docs:
        cleaned.append(' '.join(lemmatizer.lemmatize(word.lower()) for word in doc.split()
                               if letters_only(word) and word.lower() not in all_names))
        return cleaned
cleaned_emails = clean_test(emails)
#print(cleaned_emails[0])

#Vectorize the emails
cv = CountVectorizer(stop_words="english", max_features=500)
term_docs = cv.fit_transform(cleaned_emails)
feauture_names = cv.get_feature_names()
#print(term_docs)

#group the data by label
def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index

label_index = get_label_index(labels)

#print(label_index)
#print(labels)

#Compute prior
def get_prior(label_index):
    prior = {label: len(index) for label, index in label_index.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior

#Calculate likelihood based out training samples
def get_likelihood(term_document_matrix, label_index, smoothing = 0):
    likelihood = {}
    for label, index in label_index.items():
        likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)
    return likelihood

#Calculating likelihood
smoothing=1
likelihood =get_likelihood(term_docs, label_index, smoothing)
print (likelihood[0][:5])