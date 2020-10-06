import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import requests
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from keras import models
from keras import layers

Data = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')

print(Data.head(10))
groupby_df = Data[["label","tweet"]].groupby(by='label').count()
print(groupby_df)
groupby_df.plot.bar()
plt.show(block=True)


# It ensuers that there are no more null values presented in the tweet and label column
print("------------------Checking the missing values-----------------------")
Data['tweet'].isnull()
Data['label'].isnull()
print(Data.head(10))
print("--------------------Checking null values------------------------------")
# Remove if any null values if any
modifiedDF=Data.fillna(" ")

# verify no longer null values presented in the data
print(modifiedDF.isnull().sum())
print("-------------------------------------------------------------------------------")
stopwords = stopwords.words("english")

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector



def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stopwords]

    snowball = SnowballStemmer("english")
    stemmed_words = [snowball.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

    return " ".join(lemma_words)


from nltk.tokenize import TweetTokenizer
modifiedDF.tweet = modifiedDF['tweet'].apply(preprocess_tweet_text)
print(modifiedDF)
print(modifiedDF.shape)

modifiedDF = modifiedDF[['tweet','label']]
print(modifiedDF.shape)


from keras.preprocessing.text import Tokenizer
import collections
tk = Tokenizer(num_words=10000)
tk.fit_on_texts(modifiedDF.tweet)
X=tk.texts_to_matrix(modifiedDF.tweet)
print(X.shape)

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

le = LabelEncoder()
Y= le.fit_transform(modifiedDF.label)

print(X.shape[0])
no_epochs=10

base_model = models.Sequential()
base_model.add(layers.Dense(64, activation='relu',input_shape=(10000,)))
base_model.add(layers.Dense(32, activation='relu'))
base_model.add(layers.Dense(16, activation='relu'))
base_model.add(layers.Dense(2, activation='softmax'))
base_model.summary()

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, random_state=37)

def deep_model(model):
    model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train , y_train , epochs=no_epochs, batch_size=32 , validation_data=(X_test,y_test), verbose=0)
    return history


def eval_metric(history, metric_name):
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]
    e = range(1, no_epochs + 1)
    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.legend()
    plt.show()


base_history = deep_model(base_model)
train_loss,training_accuracy=base_model.evaluate(X_train,y_train)
test_loss,testing_accuracy=base_model.evaluate(X_test, y_test)
print("Training Loss={},Accuracy={}".format(train_loss,training_accuracy))
print("Testing Loss={},Accuracy={}".format(test_loss,testing_accuracy))

# make class predictions with the model
predictions = base_model.predict_classes(X_test)
# summarize the first 5 cases
for i in range(len(modifiedDF)):
    print(' predicted = %d (Ground Truth %d)' % (predictions[i], y_test[i]))

eval_metric(base_history, 'loss')
eval_metric(base_history,'accuracy')

