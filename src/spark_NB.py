import sys

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer, HashingTF, Tokenizer, RegexTokenizer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer

sc = SparkContext.getOrCreate()

spark = SparkSession(sc)


def read_data(byte_data_directory, x_filename, y_filename=None):
    """
    reads in byte date from a list of filenames given in file located
    at x_filename. if y_filename is supplied labels will be read in and a map
    will be created as well and a label column added to the returned dataframe
    """

    xfile = open(x_filename)
    X_files = xfile.read().splitlines()

    X_filenames = list(map(lambda x: byte_data_directory+x+'.bytes', X_files))
    dat = sc.wholeTextFiles(",".join(X_filenames))

    if(y_filename is not None):
        yfile = open(y_filename)
        y_labels = yfile.read().splitlines()
        label_map = sc.broadcast(dict(zip(X_filenames, y_labels)))
        dat = dat.map(lambda x: (x[0], x[1], float(label_map.value[x[0]])))
        dat = dat.toDF(['filname', 'text', 'category']).repartition(12)
    else:
        dat = dat.toDF(['filname', 'text']).repartition(12)

    return(dat)


def create_pipeline():
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
    nb = NaiveBayes(smoothing=1)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, label_stringIdx, nb])

    return(pipeline)


dat_train = read_data(sys.argv[5],
                      sys.argv[1], sys.argv[2])
pipeline = create_pipeline()

# fit the pipeline to the training data
model = pipeline.fit(dat_train)

dat_test = read_data(sys.argv[5],
                     sys.argv[3], sys.argv[4])

# create predictions on testing set
pred = model.transform(dat_test)
pred.show()

# evaluate model on texting set predictions
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print(evaluator.evaluate(pred))
