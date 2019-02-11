from pyspark import SparkContext,SparkConf
from pyspark import SparkContext,SparkConf
from pyspark.sql.types import *
from pyspark.sql.functions import udf,col,split
from pyspark.sql import SQLContext
from pyspark.ml.feature import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import concat, col, lit
import requests
import os, sys
import argparse
import re
import pyspark


def fetch_url(file_row):
    path = 'https://storage.googleapis.com/uga-dsp/project1/data/bytes'
    fetch_url = path+"/"+ file_row + ".bytes"

    #Get the file data and convert into text.
    byte_text = requests.get(fetch_url).text
    return(byte_text)


def preprocessing(data_row):

    #Removing the blank lines.
    data_row = re.sub('\\r\\n', ' ', str(data_row))

    #Removing the pair of unnecessary question-marks.
    data_row = re.sub('\??', '', str(data_row))

    #Removing the rows of Cs.
    data_row = re.sub('CC CC CC CC CC CC CC CC', '', str(data_row))

    #Remove the headers. Words larger than 2 characters.
    data_row = re.sub(r'\b[A-Z|0-9]{3,}\b', '', str(data_row))

    #Remove Multiple Spaces to one.
    data_row = re.sub(r' +', ' ', str(data_row))

    #Strip the text
    data_row = data_row.strip()
    return data_row

conf = SparkConf().setAppName('P2MalewareClassification')
conf = (conf.setMaster('local[*]')).set('spark.executor.memory', '20G').set('spark.driver.memory','25G').set('spark.driver.maxResultSize','13G')

conf =conf.set('spark.driver.memory', '25g')
conf =conf.set('spark.executor.memory', '20g')
#conf =conf.set('spark.driver.cores', '4')
#conf =conf.set('spark.executor.cores', '6')
#conf =conf.set('spark.python.worker.memory', '4g')
#conf =conf.set('spark.yarn.am.memoryOverhead', '1g')
#conf =conf.set('spark.yarn.driver.memoryOverhead', '2g')
#conf =conf.set('spark.yarn.executor.memoryOverhead', '2g')
#conf =conf.set('spark.driver.maxResultSize', '6g')

#Create the spark context.
sc = SparkContext.getOrCreate(conf=conf)
#Create SQL context.
sqlContext = SQLContext(sc)

#Get the training file names and make zip index as key.
rdd_train_x = sc.textFile("gs://uga-dsp/project1/files/X_small_train.txt").zipWithIndex().map(lambda l:(l[1],l[0]))

#Get the training lables and make zip index as key.
rdd_train_y = sc.textFile("gs://uga-dsp/project1/files/y_small_train.txt").zipWithIndex().map(lambda l:(l[1],l[0]))

#Get the testing file names and make zip index as key.
rdd_test_x = sc.textFile("gs://uga-dsp/project1/files/X_small_test.txt").zipWithIndex().map(lambda l:(l[1],l[0]))
rdd_test_y = sc.textFile("gs://uga-dsp/project1/files/y_small_test.txt").zipWithIndex().map(lambda l:(l[1], l[0]))

#Join training by index to create a merged set.
rdd_train = rdd_train_x.join(rdd_train_y)
rdd_test = rdd_test_x.join(rdd_test_y)

#Get the  file list files.
#Making the pair of (Label, Bytes)
rdd_train_text = rdd_train.map(lambda x: (x[0], x[1][1], fetch_url(x[1][0]))).map(lambda x: (x[0], x[1], preprocessing(x[2])))
rdd_test_text = rdd_test.map(lambda x: (x[0], x[1][1], fetch_url(x[1][0]))).map(lambda x: (x[0], x[1], preprocessing(x[2])))

#Collect the data.
#df_train_text = rdd_train_text.toDF("category", "text")
#df_train_text.show()
#rdd_test_text.collect()
print("Data fetched successful.")

df_train_original = sqlContext.createDataFrame(rdd_train_text, schema=["index","category", "text"])
df_train_original.show()
df_test_original = sqlContext.createDataFrame(rdd_test_text, schema=["index", "category", "text"])
