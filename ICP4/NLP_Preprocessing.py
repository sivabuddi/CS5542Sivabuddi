import pandas as pd
import numpy as np
import re
import collections
import matplotlib.pyplot as plt

# Packages for data preparation
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder


import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, ImageColorGenerator
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.svm import SVC, LinearSVC

# Packages for modeling
from keras import models
from keras import layers
from keras import regularizers


def convert_list_to_string(org_list, seperator=' '):
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    return seperator.join(org_list)


df = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')
print("---original df---")
print(df.head(10))
groupby_df = df[["label","tweet"]].groupby(by='label').count()
print(groupby_df)
groupby_df.plot.bar()
plt.show(block=True)
# It ensuers that there are no more null values presented in the tweet and label column
print("------------------Checking the missing values---------------------------------")
print(df['tweet'].isnull())
print(df['label'].isnull())
print("--------------------Checking null values--------------------------------------")
# Remove if any null values if any
modifiedDF=df.fillna(" ")
# verify no longer null values presented in the df
print(modifiedDF.isnull().sum())
print("-------------------------------------------------------------------------------")

# Read the tweet text only  :
text = pd.Series(modifiedDF.tweet.head(100)).to_string()
#Tokenize the text by sentences :
sentences = sent_tokenize(text)
#How many sentences are there? :
print ("number of sentences",len(sentences))
#Tokenize the text with words :
words = word_tokenize(text)
#Print words :
print(words)
#How many words are there? :
print("number of words",len(words))
print("\n")

# Preprocessing the df which includes removing puncuations,handers, numbers, special characters, stop words
#Empty list to store words:
words_no_punc = []

#Removing punctuation marks, handler @,numbers, special characters, stop words :
for w in words:
    if w.isalpha():
        words_no_punc.append(w.lower())
#Print the words without punctution marks :
print(words_no_punc)

print("---------------------------------Before removing stop words frequency distribution---------------------\n")
fdist= FreqDist(words_no_punc)
print(fdist.most_common(10))
#Plot the graph for words_no_punc using fdist :
import matplotlib.pyplot as plt
print(fdist.plot(10,title="Before removing Stop words"))
#
print("---------------------------------After removing stop words frequency distribution---------------------\n")
stopwords = stopwords.words("english")
# Empty list to store clean words :
clean_words = []
for w in words_no_punc:
    if w not in stopwords:
        clean_words.append(w)

fdist = FreqDist(clean_words)
print(fdist.most_common(10))
print(clean_words)
print("Before eliminating stop words, total no.of.tokens",len(words_no_punc))
print("After eliminating stop words, total no.of.tokens",len(clean_words))
#
#Frequency distribution :
fdist = FreqDist(clean_words)
fdist.most_common(10)

#Plot the most common words on grpah:
fdist.plot(10,title="After removing Stop words")

# Performing Stemming
snowball = SnowballStemmer("english")
#Word-list for stemming :
print("-------------------------------------Stemming---------------------------------------")
print("{0:20}{1:20}".format("Word","Snowball Stemmer"))


for w in clean_words:
    print("{0:20}{1:20}".format(w,snowball.stem(w)))
#
print(type(clean_words))
# Convert list of strings to string
clean_words_str = convert_list_to_string(clean_words)
print(type(clean_words_str))
print(clean_words_str)

# PoS tagging
# Tokenizing words :
tokenized_words = word_tokenize(clean_words_str)

for words in tokenized_words:
    tagged_words = nltk.pos_tag(tokenized_words)

print("-------------------------------------POS Tagging---------------------------------------")
print(tagged_words)


#Create an object :
cv = CountVectorizer()

#Generating output for Bag of Words :
B_O_W = cv.fit_transform(clean_words).toarray()

#Features :
# print(cv.get_feature_names())
# print("\n")
print("---------------------------------Building Bag-of-Words-------------------------------")
#Show the output :
print(B_O_W)
print(cv.get_feature_names())
#
print("---------------------------------Building TF-IDF Vectors-------------------------------")
#Create an object :
vectorizer = TfidfVectorizer(norm = None)

#Generating output for TF_IDF :
X = vectorizer.fit_transform(clean_words).toarray()

#Total words with their index in model :
# print(vectorizer.vocabulary_)
# print("\n")


#Show the output :
print(X)

#Features :
print(vectorizer.get_feature_names())
print("\n")

