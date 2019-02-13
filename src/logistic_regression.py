from operator import add
import json
import sys
from math import log
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes, LogisticRegression
from pyspark.ml.feature import CountVectorizer, HashingTF, Tokenizer, RegexTokenizer, StringIndexer
from operator import add
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import re

sc = SparkContext.getOrCreate()
spark=SparkSession(sc)
def read_data(byte_data_directory, x_filename, y_filename=None):
    """
    reads in byte date from a list of filenames given in file located
    at x_filename. if y_filename is supplied labels will be read in and a map
    will be created as well and a label column added to the returned dataframe
    """
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
    model pipeline
    Currently uses RegexTokenizer to get bytewords as tokens, hashingTF to
    featurize the tokens as word counts, and Logistic Regression to fit and classify
    This is where most of the work will be done in improving the model
    """

label_stringIdx = StringIndexer(inputCol="category", outputCol="label")
tokenizer = RegexTokenizer(inputCol="text", outputCol="words",
                               pattern="(?<=\\s)..", gaps=False)
hashingTF = HashingTF(numFeatures=256, inputCol=tokenizer.getOutputCol(),
                          outputCol="features")
lr = LogisticRegression(maxIter=1, featuresCol="features", labelCol="label", family="multinomial")

pipeline = Pipeline(stages=[tokenizer, hashingTF, label_stringIdx, lr])

testLabels = sys.argv[4] if sys.argv[4] != 'None' else None


dat_train = read_data(sys.argv[5],
                      sys.argv[1], sys.argv[2])


# The Pipeline is treated as an Estimator, wrapping it in a CrossValidator instance.
# This will allow us to jointly choose parameters for all Pipeline stages.
# A set of Estimator ParamMaps and an Evaluator are required by a CrossValidator. 
# A ParamGridBuilder is used to construct a grid of parameters to search over.
# With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
# this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 100, 256]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=3)  

#fitting using cross validation to find the best model 
model = crossval.fit(dat_train)

dat_test = read_data(sys.argv[5],
                      sys.argv[3], testLabels)

# create predictions on testing set
pred = model.transform(dat_test)
pred.persist()
pred.show()


#Model evaluation
evaluator=MulticlassClassificationEvaluator(labelCol="category", predictionCol="prediction", metricName="accuracy")
accuracy=evaluator.evaluate(pred)
print("Accuracy=%g" % (accuracy))
print("Test Error=%g" % (1.0-accuracy))
