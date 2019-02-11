import sys

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer, HashingTF, RegexTokenizer
from pyspark.ml.feature import NGram
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F


sc = SparkContext.getOrCreate()

spark = SparkSession(sc)


def read_data(byte_data_directory, x_filename, y_filename=None):
    """
    reads in byte date from a list of filenames given in file located
    at x_filename. if y_filename is supplied labels will be read in and a map
    will be created as well and a label column added to the returned dataframe
    """

    X_files = sc.textFile(x_filename).collect()

    X_filenames = list(map(lambda x: byte_data_directory+x+'.bytes', X_files))
    dat = sc.wholeTextFiles(",".join(X_filenames), minPartitions=300)
    X_df = sc.parallelize(X_filenames, numSlices=300).map(lambda x: (x,x.split('/')[-1])).toDF(['filename','byte'])
    X_df = X_df.withColumn("idx", F.monotonically_increasing_id())

    if(y_filename is not None):
        y_labels = sc.textFile(y_filename).collect()
        label_map = sc.broadcast(dict(zip(X_filenames, y_labels)))
        dat = dat.map(lambda x: (x[0], x[1], int(label_map.value[x[0]])))
        dat = dat.toDF(['filename', 'text', 'label'])
    else:
        dat = dat.toDF(['filename', 'text'])
        dat.show()
    dat = X_df.join(dat, X_df.filename == dat.filename, how='left').sort("idx")
    print(dat.rdd.getNumPartitions())
    return(dat)


def create_pipeline():
    """
    creates model pipeline
    Currently uses RegexTokenizer to get bytewords as tokens, hashingTF to
    featurize the tokens as word counts, and NaiveBayes to fit and classify

    This is where most of the work will be done in improving the model
    """

    tokenizer = RegexTokenizer(inputCol="text", outputCol="words",
                               pattern="(?<=\\s)..", gaps=False)
    ngram = NGram(n=2, inputCol="words", outputCol="grams")
    hashingTF = HashingTF(numFeatures=65792, inputCol=ngram.getOutputCol(),
                          outputCol="features")
    nb = NaiveBayes(smoothing=1)
    pipeline = Pipeline(stages=[tokenizer, ngram, hashingTF, nb])

    return(pipeline)


testLabels = sys.argv[4] if sys.argv[4] != 'None' else None


dat_train = read_data(sys.argv[5],
                      sys.argv[1], sys.argv[2])

pipeline = create_pipeline()

# fit the pipeline to the training data
model = pipeline.fit(dat_train)

dat_test = read_data(sys.argv[5],
                     sys.argv[3], testLabels)
dat_test.show()
# create predictions on testing set
pred = model.transform(dat_test)
pred.persist()
pred.show()

pred_list = pred.select("prediction").rdd.map(lambda x: int(x[0]+1)).collect()
print(pred_list)
