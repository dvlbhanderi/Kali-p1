from operator import add
import json
import sys
from math import log
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes, LogisticRegression
#from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import CountVectorizer, HashingTF, Tokenizer, RegexTokenizer
from operator import add
import re
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
"""
    reads in byte date from a list of filenames given in file located
    at x_filename. if y_filename is supplied labels will be read in and a map
    will be created as well and a label column added to the returned dataframe
    """

sc = SparkContext.getOrCreate()
spark=SparkSession(sc)
def read_data(byte_data_directory, x_filename, y_filename=None):
    
    xfile = sc.textFile(x_filename)
    def func(x):
        x=x.split()
        x[0]=x[0].encode("ascii","replace")
        return(x)
    X_files = xfile.flatMap(func)
    X_filenames = list(map(lambda x: byte_data_directory+x+'.bytes', X_files.collect()))
    dat = sc.wholeTextFiles(",".join(X_filenames))

    if(y_filename is not None):
        yfile = sc.textFile(y_filename)
        y_labels = yfile.flatMap(func)
        label_map = sc.broadcast(dict(zip(X_filenames, y_labels.collect())))
        dat = dat.map(lambda x: (x[0], x[1], float(label_map.value[x[0]])))
        dat = dat.toDF(['filname', 'text', 'category']).repartition(12)
    else:
        dat = dat.toDF(['filname', 'text']).repartition(12)

    return(dat)


    """
    creates model pipeline
    Currently uses RegexTokenizer to get bytewords as tokens, hashingTF to
    featurize the tokens as word counts, and NaiveBayes to fit and classify
    This is where most of the work will be done in improving the model
    """

label_stringIdx = StringIndexer(inputCol="category", outputCol="label")
tokenizer = RegexTokenizer(inputCol="text", outputCol="words",
                               pattern="(?<=\\s)..", gaps=False)
hashingTF = HashingTF(numFeatures=256, inputCol=tokenizer.getOutputCol(),
                          outputCol="features")
lr = LogisticRegression(maxIter=1, featuresCol="features", labelCol="label", family="multinomial")
pipeline = Pipeline(stages=[tokenizer, hashingTF, label_stringIdx, lr])



dat_train = read_data('gs://uga-dsp/project1/data/bytes/',
                      'gs://black_bucket/X_small_train.txt', 'gs://black_bucket/y_small_train.txt')

paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=3)  # use 3+ folds in practice
model = crossval.fit(dat_train)

dat_test = read_data('gs://uga-dsp/project1/data/bytes/',
                      'gs://black_bucket/X_small_test.txt', 'gs://black_bucket/y_small_test.txt')

# create predictions on testing set
pred = model.transform(dat_test)
pred.persist()
pred.show()
pred.select("prediction").distinct().show()
pred.select("label").distinct().show()
pred.select("category").distinct().show()
evaluator=MulticlassClassificationEvaluator(labelCol="category", predictionCol="prediction", metricName="accuracy")
accuracy=evaluator.evaluate(pred)
print("Accuracy=%g" % (accuracy))
print("Test Error=%g" % (1.0-accuracy))


#spark.yarn.executor.memoryoverhead
#spark.driver.maxResultSize
#spark.driver.memory
#spark.executor.memory
#pred.select("prediction").withColumn("pred1",col("prediction").cast(StringType()).select("pred1").coalesce(1).write.text('gs://black_bucket/model_out')
#rdd().map(lambda x: int(x+1))
#data = sc.wholeTextFiles("/home/rutu/DSP/a7Niv6pD5WPycnQK1TZ8.bytes")
