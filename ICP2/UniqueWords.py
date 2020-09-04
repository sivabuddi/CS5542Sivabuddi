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

sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)


def removePunctuation(column):
    return trim(lower(regexp_replace(column, '([^\s\w_a-zA-Z\[0-9]]|_)+', ''))).alias('sentence')
    #return trim(lower(regexp_replace(column, '([^\s\w_]|_)+', ''))).alias('sentence')



def wordCount(wordListDF):
    return wordListDF.groupBy('word').count()

# Display the type of the Spark sqlContext
print(type(sqlContext))

fileName = "icp2.txt"

DF_New = sqlContext.read.text(fileName).select(removePunctuation(col('value')))
DF_New.show(truncate=False)

shakeWordsSplitDF = (DF_New.select(split(DF_New.sentence, '\s+').alias('split')))
shakeWordsSingleDF = (shakeWordsSplitDF.select(explode(shakeWordsSplitDF.split).alias('word')))
shakeWordsDF = shakeWordsSingleDF.where(shakeWordsSingleDF.word != '')
print("shape of the data: ({},{})".format(shakeWordsDF.count(),len(shakeWordsDF.dtypes)))


# Count the words
from pyspark.sql.functions import desc
WordsAndCountsDF = wordCount(shakeWordsDF)
topWordsAndCountsDF = WordsAndCountsDF.orderBy("word", ascending=False)
topWordsAndCountsDF.show(n=200)


# Count the unique words
WordsAndCountsDF = wordCount(shakeWordsDF)
unique_words = WordsAndCountsDF.select(countDistinct("word").alias("UniqueWords")).show()

# shakeWordsDF['word'].str.get(0)




# # print(type(shakeWordsDF))
# shakeWordsDFCount = shakeWordsDF.count()
# print(shakeWordsDFCount)
#
# # Count the words
# from pyspark.sql.functions import desc
#
# WordsAndCountsDF = wordCount(shakeWordsDF)
# topWordsAndCountsDF = WordsAndCountsDF.orderBy("word", ascending=True)
# topWordsAndCountsDF.show(n=200)
#


























