from pyspark import SparkContext, SparkConf
import shutil
import os

spark_conf = SparkConf().setAppName("Big Data Analytics ICP2").setMaster("local[*]")
spark_context = SparkContext(conf=spark_conf)


def process_group_by(output="group_by/", lower=False):
    """
    function : process_group_by
    description: the function capable to take inputs as output directory and the case of the letter for processing.
    this uses spark reduce by key process to write values into output directory.
    """
    content_rdd = spark_context.textFile("icp2.txt")
    nonempty_lines = content_rdd.filter(lambda x: len(x) > 0)
    if lower:
        words = nonempty_lines.flatMap(lambda x: x.lower().split(" "))
    else:
        words = nonempty_lines.flatMap(lambda x: x.split(" "))
    print(words.count())
    result = words.map(lambda x: (x[0], x)).groupByKey().map(lambda x: (x[0], list(x[1])))
    result.coalesce(1, True).saveAsTextFile(output)
    for each in result.collect():
        print(each)


def process_reduce_by(output="reduce_by/", lower=False, as_string=False):
    """
    function : process_group_by
    description: the function capable to take inputs as output directory and the case of the letter for processing.
    this uses spark reduce by key process to write values into output directory.
    """
    content_rdd = spark_context.textFile("icp2.txt")
    nonempty_lines = content_rdd.filter(lambda x: len(x) > 0)
    if lower:
        words = nonempty_lines.flatMap(lambda x: x.lower().split(" "))
    else:
        words = nonempty_lines.flatMap(lambda x: x.split(" "))
    print(words.count())
    if as_string:
        result = words.map(lambda x: (x[0], x)).reduceByKey(lambda x, y: x + " , " + y)
    else:
        result = words.map(lambda x: (x[0], [x])).reduceByKey(lambda x, y: x + y)
    result.coalesce(1, True).saveAsTextFile(output)
    for each in result.collect():
        print(each)


def check_delete_files():
    for each in os.listdir("."):
        if each == "sreen_prints":
            continue
        if not os.path.isfile(each):
            print(each)
            shutil.rmtree(each)


'''
main function call which can invoke the reduce by and group by methods to handle the process of the grouping values
with reference as keys.
'''
if __name__ == "__main__":
    check_delete_files()
    # process_group_by()
    # process_reduce_by()
    # process_group_by(output="lower_case_group_by", lower=True)
    # process_reduce_by(output="lower_case_reduce_by", lower=True)
    # process_reduce_by(output="as_string_reduce_by", as_string=True)
