from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer, HashingTF, Tokenizer, RegexTokenizer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer

sc = SparkContext.getOrCreate()

spark = SparkSession(sc)

# Read in small files
xfile = open("X_small_train.txt")
X_train = xfile.read().splitlines()
yfile = open("y_small_train.txt")
y_train = yfile.read().splitlines()

# set directory of byte data
byte_data_directory = 'file:/home/durden/Desktop/Practicum/Kali-p1/data/'

# build out list of full byte file names
X_filenames = list(map(lambda x: byte_data_directory+x+'.bytes', X_train))

# create local map of training labels
label_map = sc.broadcast(dict(zip(X_filenames, y_train)))

# pull byte data and transform to dataframe
dat_train = sc.wholeTextFiles(",".join(X_filenames))
dat_train = dat_train.map(lambda x: (x[0], x[1], float(label_map.value[x[0]])))
dat_train = dat_train.toDF(['filname', 'text', 'category']).repartition(12)

# create pipeline for labels, tokenize, featurize, and classify
label_stringIdx = StringIndexer(inputCol="category", outputCol="label")
tokenizer = RegexTokenizer(inputCol="text", outputCol="words",
                           pattern="(?<=\\s)..", gaps=False)
hashingTF = HashingTF(numFeatures=256, inputCol=tokenizer.getOutputCol(),
                      outputCol="features")
nb = NaiveBayes(smoothing=1)
pipeline = Pipeline(stages=[tokenizer, hashingTF, label_stringIdx, nb])

# fit the pipeline to the training data
model = pipeline.fit(dat_train)

# read in the testing data
xfile = open("X_small_test.txt")
X_test = xfile.read().splitlines()
yfile = open("y_small_test.txt")
y_test = yfile.read().splitlines()

# create list of testing byte files
X_filenames_test = list(map(lambda x: byte_data_directory+x+'.bytes', X_test))

# create labels for small testing dataset
label_map_test = sc.broadcast(dict(zip(X_filenames_test, y_test)))

# transform testing set to dataframe
dat_test = sc.wholeTextFiles(",".join(X_filenames_test))
dat_test = dat_test.map(lambda x: (x[0], x[1], float(label_map_test.value[x[0]])))
dat_test = dat_test.toDF(['filname', 'text', 'category']).repartition(12)

# create predictions on testing set
pred = model.transform(dat_test)
pred.select("filname", "label", "prediction").show()

# evaluate model on texting set predictions
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print(evaluator.evaluate(pred))
