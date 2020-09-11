#Stemming Example :
#Import stemming library :
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

porter = PorterStemmer()

#Word-list for stemming :
word_list = ["studies","leaves","decreases","plays"]

for w in word_list:
    print(porter.stem(w))

#Stemming Example :

#Import stemming library :
from nltk.stem import SnowballStemmer

snowball = SnowballStemmer("english")

#Word-list for stemming :
word_list = ["studies","leaves","decreases","plays"]

for w in word_list:
    print(snowball.stem(w))


#Print languages supported :
print('\n')
print(SnowballStemmer.languages)

#Lemitization Example :
from nltk import WordNetLemmatizer

lemma = WordNetLemmatizer()
word_list = ["studies","leaves","decreases","plays"]

for w in word_list:
    print(lemma.lemmatize(w ,pos="v"))

word_list1 = ["am","is","are","was","were"]
print('\n')
for w in word_list1:
    print(lemma.lemmatize(w ,pos="v"))

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('studying', pos="v"))
print(lemmatizer.lemmatize('studying', pos="n"))
print(lemmatizer.lemmatize('studying', pos="a"))
print(lemmatizer.lemmatize('studying', pos="r"))


from nltk import pos_tag
#PoS tagging :
tag = nltk.pos_tag(["Studying","Study"])
print (tag)


from nltk import  word_tokenize
# PoS tagging example :
sentence = "In today's Big Data analytics and application ICP we are learning about core NLP"

# Tokenizing words :
tokenized_words = word_tokenize(sentence)

for words in tokenized_words:
    tagged_words = nltk.pos_tag(tokenized_words)

print(tagged_words)



# parser example
#Extracting Noun Phrase from text :

# ? - optional character
# * - 0 or more repetations
grammar = "NP : {<DT>?<JJ>*<NN>} "
import matplotlib.pyplot as plt
#Creating a parser :
parser = nltk.RegexpParser(grammar)

#Parsing text :
output = parser.parse(tagged_words)
print (output)


# NER example
#Sentence for NER :
sentence = "Mr. Trump made a deal on a beach of Switzerland near WHO on 01/02/2020."

#Tokenizing words :
tokenized_words = word_tokenize(sentence)

#PoS tagging :
for w in tokenized_words:
    tagged_words = nltk.pos_tag(tokenized_words)

#print (tagged_words)

#Named Entity Recognition :
N_E_R = nltk.ne_chunk(tagged_words,binary=False)
print(N_E_R)

# WordNet example

# Import wordnet :
from nltk.corpus import wordnet

# Word meaning with definitions :
for words in wordnet.synsets("Fun"):
    print(words.name())
    print(words.definition())
    print(words.examples())

    for lemma in words.lemmas():
        print(lemma)
    print("\n")

# Finding synonyms and antonyms :

# Empty lists to store synonyms/antonynms :
synonyms = []
antonyms = []

for words in wordnet.synsets('old'):
    for lemma in words.lemmas():
        synonyms.append(lemma.name())
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())

# Print lists :
print(synonyms)
print("\n")
print(antonyms)

#Similarity in words :
word1 = wordnet.synsets("pond","n")[0]
word2 = wordnet.synsets("lake","n")[0]

#Check similarity :
print(word1.wup_similarity(word2))





# Bag of words example
#Import required libraries :
from sklearn.feature_extraction.text import CountVectorizer

#Text for analysis :
sentences = ["Lorem Ipsum is simply dummy text of the printing and typesetting industry.",
             "Contrary to popular belief, Lorem Ipsum is not simply random text. ",
             "It has roots in a piece of classical Latin literature"]

#Create an object :
cv = CountVectorizer()

#Generating output for Bag of Words :
B_O_W = cv.fit_transform(sentences).toarray()

#Total words with their index in model :
print(cv.vocabulary_)
print("\n")

#Features :
print(cv.get_feature_names())
print("\n")

#Show the output :
print(B_O_W)



# TFIDF example
#Import required libraries :
from sklearn.feature_extraction.text import TfidfVectorizer

#Sentences for analysis :
sentences = ['He is creating a template ',
             'I’m designing a document']
#Create an object :
vectorizer = TfidfVectorizer(norm = None)

#Generating output for TF_IDF :
X = vectorizer.fit_transform(sentences).toarray()

#Total words with their index in model :
print(vectorizer.vocabulary_)
print("\n")

#Features :
print(vectorizer.get_feature_names())
print("\n")

#Show the output :
print(X)