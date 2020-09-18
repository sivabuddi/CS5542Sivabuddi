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

groupby_df = Data[["label","tweet"]].groupby(by='label').count()
print(groupby_df)
groupby_df.plot.bar()
plt.show(block=True)

print("-------------------After Performing Under sampling--------------------------------------------")
zeros = Data[Data['label'] == 0]
ones = Data[Data['label'] == 1]
print(zeros)
print(ones)

df_under=zeros.sample(2*len(ones)) # down sample to the lenght of  2*spam from the length of ham data
df_under = pd.concat([df_under,ones], axis=0)

print('Random under-sampling:')
print(len(df_under))
# # Multi-column frequency count
count = df_under.groupby(['label']).count()
print(count)

groupby_df = df_under[["label","tweet"]].groupby(by='label').count()
print(groupby_df)
groupby_df.plot.bar()
plt.show(block=True)

# import seaborn as sns
# sns.countplot(x='label', data=Data)

# It ensuers that there are no more null values presented in the tweet and label column
print("------------------Checking the missing values-----------------------")
df_under['tweet'].isnull()
df_under['label'].isnull()
print(df_under.head(10))
print("--------------------Checking null values------------------------------")
# Remove if any null values if any
modifiedDF=df_under.fillna(" ")

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
    # lemmatizer = WordNetLemmatizer()
    # lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

    return " ".join(stemmed_words)



from nltk.tokenize import TweetTokenizer
modifiedDF.tweet = df_under['tweet'].apply(preprocess_tweet_text)
print(modifiedDF)
print(modifiedDF.shape)

modifiedDF = modifiedDF[['tweet','label']]
print(modifiedDF)


X_train, X_test, y_train, y_test = train_test_split(modifiedDF.tweet , modifiedDF.label, test_size=0.2, random_state=37)

from keras.preprocessing.text import Tokenizer
import collections
tk = Tokenizer(num_words=10000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tk.fit_on_texts(X_train)

print('# Train data samples:', X_train.shape[0])
print('# Test data samples:', X_test.shape[0])
print(X_train.shape[0]== y_train.shape[0])
print(X_test.shape[0] == y_test.shape[0])


X_train_seq = tk.texts_to_sequences(X_train)
X_test_seq = tk.texts_to_sequences(X_test)


#print('"{}" is converted into {}'.format(X_train[2], X_train_seq[2]))


def one_hot_seq(seqs, nb_features = 10000):
    ohs = np.zeros((len(seqs), nb_features))
    for i, s in enumerate(seqs):
        ohs[i, s] = 1.
    return ohs

X_train_oh = one_hot_seq(X_train_seq)
X_test_oh = one_hot_seq(X_test_seq)


from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.fit_transform(y_test)
y_train_oh = to_categorical(y_train_le)
y_test_oh = to_categorical(y_test_le)



X_train_rest, X_valid, y_train_rest, y_valid = train_test_split(X_train_oh, y_train_oh, test_size=0.1, random_state=37)
no_epochs=10

base_model = models.Sequential()
base_model.add(layers.Dense(64, activation='relu',input_shape=(10000,)))
base_model.add(layers.Dense(32, activation='relu'))
base_model.add(layers.Dense(16, activation='relu'))
base_model.add(layers.Dense(2, activation='softmax'))
base_model.summary()

def deep_model(model):
    model.compile(optimizer='adam' , loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_rest , y_train_rest , epochs=no_epochs, batch_size=32 , validation_data=(X_valid,y_valid), verbose=0)
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
train_loss,training_accuracy=base_model.evaluate(X_train_rest,y_train_rest)
test_loss,testing_accuracy=base_model.evaluate(X_valid, y_valid)
print("Training Loss={},Accuracy={}".format(train_loss,training_accuracy))
print("Testing Loss={},Accuracy={}".format(test_loss,testing_accuracy))


eval_metric(base_history, 'loss')
eval_metric(base_history,'accuracy')







