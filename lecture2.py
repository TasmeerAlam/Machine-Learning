from sklearn.datasets import fetch_20newsgroups
groups = fetch_20newsgroups()
print(groups.keys())
print(groups['target_names'])
print(groups.target)
import numpy as np
print(np.unique(groups.target))
print(groups.data[0])
print(groups.target[0])
print(groups.target_names[groups.target[0]])
print(len(groups.data[0]), len(groups.data[1]))
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english", max_features=500)
bag_of_words = cv.fit_transform(groups.data)
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
#print(words_freq)
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
for word, count in words_freq:
    print(word + ":", count)
    words = []
    freqs = []
for word, count in words_freq:
    words.append(word)
    freqs.append(count)
import matplotlib.pyplot as plt
# Plot histogram
plt.bar(np.arange(10), freqs[:10], align='center')
plt.xticks(np.arange(10), words[:10])
plt.ylabel('Frequency')
plt.title('Top 10 Words')
plt.show()
# Test if a token is a word
def letters_only(astr):
    return astr.isalpha()
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
cleaned = []
all_names = set(x.lower() for x in names.words())
lemmatizer = WordNetLemmatizer()
for post in groups.data:
    cleaned.extend(list(lemmatizer.lemmatize(word.lower()) for word in post.split()
                        if letters_only(word) and word.lower() not in all_names))
    cleaned_bag_of_words = cv.fit_transform(cleaned)
print(cv.get_feature_names())

