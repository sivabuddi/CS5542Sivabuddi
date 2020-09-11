# create a silly test dataframe from Python collections (lists)
import pyspark
from pyspark.sql import SQLContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import regexp_replace, trim, col, lower
from pyspark.sql.functions import desc
from pyspark.sql.functions import split, explode
from pyspark.sql.functions import countDistinct, avg, stddev
import re
import pyspark.sql.functions as f

# Create the SparkContent
sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)

# remove puncutation marks in the given text
def removePunctuation(column):
    return trim(lower(regexp_replace(column, '([^\d+\s\w_a-zA-Z\[0-9]]|_)+', ''))).alias('sentence')
    #return trim(lower(regexp_replace(column, '([^\s\w_]|_)+', ''))).alias('sentence') re.sub('[!#?,.:";]', '', data)


# no.of.words words in the given text
def wordCount(wordListDF):
    return wordListDF.groupBy('word').count()

# Display the type of the Spark sqlContext
print(type(sqlContext))

fileName = "icp2.txt"

#read the input file and remove puncutations if any
DF_New = sqlContext.read.text(fileName).select(removePunctuation(col('value')))
DF_New.show(truncate=False)

# split the input text and keep column name as sentence
shakeWordsSplitDF = (DF_New.select(split(DF_New.sentence, '\s+').alias('split')))
# provide alias name as word by replacing sentence
shakeWordsSingleDF = (shakeWordsSplitDF.select(explode(shakeWordsSplitDF.split).alias('word')))
# retreive only non empty words presented in the text
shakeWordsDF = shakeWordsSingleDF.where(shakeWordsSingleDF.word != '')
# dimension of the data frame
print("shape of the data: ({},{})".format(shakeWordsDF.count(),len(shakeWordsDF.dtypes)))


# Count the words
from pyspark.sql.functions import desc
WordsAndCountsDF = wordCount(shakeWordsDF)
# displayting data frame in the descending order 
topWordsAndCountsDF = WordsAndCountsDF.orderBy("word", ascending=False)
topWordsAndCountsDF.show(n=200)


# Count the unique words
WordsAndCountsDF = wordCount(shakeWordsDF)
# displaying no.of.unique word count
unique_words = WordsAndCountsDF.select(countDistinct("word").alias("UniqueWords")).show()
































