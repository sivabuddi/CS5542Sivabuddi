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
import re
import string
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from keras import models
from keras import layers

Data = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')
# It ensuers that there are no more null values presented in the tweet and label column
print("------------------Checking the missing values-----------------------")
print(Data['tweet'].isnull())
print(Data['label'].isnull())
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
    # lemmatizer = WordNetLemmatizer()
    # lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

    return " ".join(stemmed_words)



from nltk.tokenize import TweetTokenizer
modifiedDF.tweet = modifiedDF['tweet'].apply(preprocess_tweet_text)
print(modifiedDF)

modifiedDF = modifiedDF[['tweet','label']]
print(modifiedDF)


X_train, X_test, y_train, y_test = train_test_split(modifiedDF.tweet , modifiedDF.label, test_size=0.2, random_state=37)
print('# Train data samples:', X_train.shape[0])
print('# Test data samples:', X_test.shape[0])
# assert X_train.shape[0] == y_train.shape[0]
# assert X_test.shape[0] == y_test.shape[0]
print(X_train.shape[0]== y_train.shape[0])
print(X_test.shape[0] == y_test.shape[0])






# Converting words to numbers
'''
To use the text as input for a model, we first need to convert the tweet's words into tokens, which simply means converting the words to integers that 
refer to an index in a dictionary. Here we will only keep the most frequent words in the train set. We clean up the text by applying filters and putting 
the words to lowercase. Words are separated by spaces.


'''
from keras.preprocessing.text import Tokenizer
import collections
tk = Tokenizer(num_words=50000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tk.fit_on_texts(X_train)

print('Fitted tokenizer on {} documents'.format(tk.document_count))
print('{} words in dictionary'.format(tk.num_words))
print('Top 5 most common words are:', collections.Counter(tk.word_counts).most_common(5))

'''
After having created the dictionary we can convert the text to a list of integer indexes.
This is done with the text_to_sequences method of the Tokenizer.
'''

X_train_seq = tk.texts_to_sequences(X_train)
X_test_seq = tk.texts_to_sequences(X_test)


print('"{}" is converted into {}'.format(X_train[2], X_train_seq[2]))

def one_hot_seq(seqs, nb_features = 50000):
    ohs = np.zeros((len(seqs), nb_features))
    for i, s in enumerate(seqs):
        ohs[i, s] = 1.
    return ohs

X_train_oh = one_hot_seq(X_train_seq)
X_test_oh = one_hot_seq(X_test_seq)




# print('"{}" is converted into {}'.format(X_train_seq[2], X_train_oh[2]))
# print('For this example we have {} features with a value of 1.'.format(X_train_oh[2].sum()))

'''
Converting the target classes to numbers
We need to convert the target classes to numbers as well, which in turn are one-hot-encoded with the to_categorical method in keras.

'''
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)
y_train_oh = to_categorical(y_train_le)
y_test_oh = to_categorical(y_test_le)



# print('"{}" is converted into {}'.format(y_train[0], y_train_le[0]))
# print('"{}" is converted into {}'.format(y_train_le[0], y_train_oh[0]))


'''
Splitting of a validation set
Now that our data is ready, we split of a validation set. This validation set will be used to evaluate the model performance
when we tune the parameters of the model.
'''

# tf_vector = get_feature_vector(np.array(modifiedDF.iloc[:, 2]).ravel())
# X = tf_vector.transform(np.array(modifiedDF.iloc[:, 2]).ravel())
# y = np.array(modifiedDF.iloc[:, 1]).ravel()

X_train_rest, X_valid, y_train_rest, y_valid = train_test_split(X_train_oh, y_train_oh, test_size=0.1, random_state=37)

# assert X_valid.shape[0] == y_valid.shape[0]
# assert X_train_rest.shape[0] == y_train_rest.shape[0]

print(X_valid.shape[0] == y_valid.shape[0])
print(X_train_rest.shape[0] == y_train_rest.shape[0])
print('Shape of validation set:',X_valid.shape)

# Deep learning
'''
We start with a model with 2 densely connected layers of 64 hidden elements. The input_shape for the first layer is equal to the number of words we allowed 
in the dictionary and for which we created one-hot-encoded features.
As we need to predict 2 different sentiment classes, the last layer has 3 hidden elements. The softmax activation function 
makes sure the three probabilities sum up to 1.
'''
base_model = models.Sequential()
base_model.add(layers.Dense(64, activation='relu', input_shape=(50000,)))
base_model.add(layers.Dense(64, activation='relu'))
base_model.add(layers.Dense(2, activation='softmax'))
base_model.summary()


def deep_model(model):
    model.compile(optimizer='rmsprop' , loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_rest , y_train_rest  , epochs=10, batch_size=32 , validation_data=(X_valid, y_valid) , verbose=0)
    return history


def eval_metric(history, metric_name):
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]
    e = range(1, 10 + 1)
    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.legend()
    plt.show()

base_history = deep_model(base_model)

# y_predicted=base_model.predict(X_test)
# print(y_predicted.shape)
# print(y_valid.shape)


eval_metric(base_history, 'loss')
eval_metric(base_history,'accuracy')



# from sklearn.metrics import confusion_matrix,classification_report
# # confusion matrix
# matrix = confusion_matrix(y_test,y_predicted)
# print(matrix)
# report=classification_report(y_test,y_predicted)
# print(report)
