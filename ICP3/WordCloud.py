#Import required libraries :
import pandas as pd

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize


import nltk
nltk.download("popular")

Data = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')
# Setting to display All rows of Dataframe
# pd.set_option('display.max_rows', None)
#Setting to display All Columns in Dataframe
pd.set_option('display.max_columns', None)
# Setting to display Dataframe with full width i.e. all columns in a line
pd.set_option('display.width', None)
# Setting to display Dataframe by maximizing column width
pd.set_option('display.max_colwidth', -1)
print(Data)

# Read the tweet text only  :
text = pd.Series(Data.tweet.head(50)).to_string()

#Tokenize the text by sentences :
sentences = sent_tokenize(text)

#How many sentences are there? :
print ("number of sentences",len(sentences))

#Print the sentences :
print(sentences)

#Tokenize the text with words :
words = word_tokenize(text)

#Print words :
print (words)

#How many words are there? :
print ("number of words",len(words))
print("\n")


#Import required libraries :
from nltk.probability import FreqDist

#Find the frequency :
fdist = FreqDist(words)

#Print 10 most common words :
print(fdist.most_common(10))

#Plot the graph for fdist :
import matplotlib.pyplot as plt

print(fdist.plot(10))


#Empty list to store words:
words_no_punc = []

#Removing punctuation marks :
for w in words:
    if w.isalpha():
        words_no_punc.append(w.lower())

#Print the words without punctution marks :
print(words_no_punc)

print ("\n")

#Length :
print(len(words_no_punc))

#Find the frequency :
fdist = FreqDist(words_no_punc)

#Print 10 most common words :
print(fdist.most_common(10))
print(fdist.plot(10))


from nltk.corpus import stopwords

#List of stopwords
stopwords = stopwords.words("english")
print("total number of stop words",len(stopwords))
print(stopwords)

# Empty list to store clean words :
clean_words = []

for w in words_no_punc:
    if w not in stopwords:
        clean_words.append(w)

print(clean_words)
print("\n")
print(len(clean_words))

#Frequency distribution :
fdist = FreqDist(clean_words)
fdist.most_common(10)

#Plot the most common words on grpah:
fdist.plot(10)


#Library to form wordcloud :
from wordcloud import WordCloud, ImageColorGenerator
import requests
import numpy as np
#Library to plot the wordcloud :
import matplotlib.pyplot as plt
from PIL import Image

# combining the image with the dataset
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

# We use the ImageColorGenerator library from Wordcloud
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)



#Generating the wordcloud :
wordcloud = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(text)

#Plot the wordcloud :
plt.figure(figsize = (12, 12))
plt.imshow(wordcloud)

#To remove the axis value :
plt.axis("off")
plt.show()