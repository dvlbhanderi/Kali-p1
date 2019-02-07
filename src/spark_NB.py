from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer, HashingTF, Tokenizer, RegexTokenizer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer

sc = SparkContext.getOrCreate()

spark = SparkSession(sc)

xfile = open("X_small_train.txt")
X_train = xfile.read().splitlines()
yfile = open("y_small_train.txt")
y_train = yfile.read().splitlines()

byte_data_directory = 'file:/home/durden/Desktop/Practicum/Kali-p1/data/'

X_filenames = list(map(lambda x: byte_data_directory+x+'.bytes', X_train))

label_map = sc.broadcast(dict(zip(X_filenames, y_train)))

dat_train = sc.wholeTextFiles(",".join(X_filenames))
dat_train = dat_train.map(lambda x: (x[0], x[1], float(label_map.value[x[0]])))
dat_train = dat_train.toDF(['filname', 'text', 'category']).repartition(12)

label_stringIdx = StringIndexer(inputCol="category", outputCol="label")
tokenizer = RegexTokenizer(inputCol="text", outputCol="words",
                           pattern="(?<=\\s)..", gaps=False)
hashingTF = HashingTF(numFeatures=256, inputCol=tokenizer.getOutputCol(),
                      outputCol="features")
nb = NaiveBayes(smoothing=1)
pipeline = Pipeline(stages=[tokenizer, hashingTF, label_stringIdx, nb])

model = pipeline.fit(dat_train)

xfile = open("X_small_test.txt")
X_test = xfile.read().splitlines()
yfile = open("y_small_test.txt")
y_test = yfile.read().splitlines()

X_filenames_test = list(map(lambda x: byte_data_directory+x+'.bytes', X_test))

label_map_test = sc.broadcast(dict(zip(X_filenames_test, y_test)))

dat_test = sc.wholeTextFiles(",".join(X_filenames_test))
dat_test = dat_test.map(lambda x: (x[0], x[1], float(label_map_test.value[x[0]])))
dat_test = dat_test.toDF(['filname', 'text', 'category']).repartition(12)

pred = model.transform(dat_test)
pred.select("filname", "label", "prediction").show()

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print(evaluator.evaluate(pred))
