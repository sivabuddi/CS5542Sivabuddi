from pyspark.conf import SparkConf
from pyspark.context import SparkContext

#Creating Basic RDD (pyspark.rdd.RDD) using SparkConf with the help of master
spark_conf = SparkConf().setAppName("BigData Siva ICP2").setMaster("local[*]")
spark_context = SparkContext(conf=spark_conf)

content_rdd_text = spark_context.textFile("icp2.txt")
print(type(content_rdd_text))
nonempty_lines = content_rdd_text.filter(lambda x: len(x) > 0)
#  Similar to map, it returns a new RDD by applying  a function to each element of the RDD, but output is flattened.
lower_words = nonempty_lines.flatMap(lambda x: x.lower().split(" "))
print(lower_words.count())
# It returns a new RDD by applying a function to each element of the RDD.   Function in map can return only one item.
result_groupby = lower_words.map(lambda x: (x[0], x)).groupByKey().map(lambda x: (x[0], list(x[1])))
print("--------------------Using groupby-----------------------------------------------------")
# Retreive the elements one by one using foreach loop using spark action called collect
for each_groupby in result_groupby.collect():
    print(each_groupby)
result_reducedby = lower_words.map(lambda x: (x[0], x)).reduceByKey(lambda x, y: x + " , " + y)
print("---------------------------------------------------------------------------------------")
print(lower_words.count())
print("--------------------Using reducedby----------------------------------------------------")
# Retreive the elements one by one using foreach loop
for each_reducedby in result_reducedby.collect():
    print(each_reducedby)

