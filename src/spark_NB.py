import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer

sc = SparkContext.getOrCreate()

spark = SparkSession(sc)


X_train = sc.textFile(sys.argv[1])
y_train = sc.textFile(sys.argv[2])

"""
X_train = sc.parallelize(['DvdM5Zpx96qKuN3cAt1y','5QpgRV2cqU9wvjBist1a','2F6ZfVCQRi3vrwcj4zxL','1KjZ4An78sOytkzgRL0E'])
y_train = sc.parallelize([6,3,1,7])
"""
print(y_train.count())
print('short files read')

# read directory of byte files
byte_data_directory = sys.argv[3]

# form full filename from directory and names and loas into list
X_filenames = X_train.map(lambda x: byte_data_directory+x+'.bytes')
X_filenames = X_filenames.collect()
label_map = dict(zip(X_filenames, [int(i) for i in y_train.collect()))
label_map = sc.broadcast(label_map)
print('dat read')
# load pairRDD of text to preproc and train on
dat_train = sc.wholeTextFiles(",".join(X_filenames))
print(dat_train.count())
dat_train = dat_train.map(lambda x: (x[0], x[1], label_map.value[x[0]]))
y_train.unpersist()

dat_train = dat_train.map(lambda x: (x[0], x[1].split(), x[2]))
dat_train = dat_train.map(lambda x: (x[0], list(filter(lambda y: len(y) == 2, x[1])), x[2]))
dat_df = dat_train.toDF(['filename', 'words', 'label'])
dat_train.unpersist()

print(dat_df.rdd.getNumPartitions())
dat_df = dat_df.repartition(32)
print('repartitioned', dat_df.rdd.getNumPartitions())


cv = CountVectorizer(inputCol="words", outputCol="features")
dat_df = cv.fit(dat_df).transform(dat_df)
print(dat_df.show())

nb = NaiveBayes(smoothing=1)
model = nb.fit(dat_df)
"""
del dat_df


# read in the testing data using same process as above
X_test = sc.textFile(sys.argv[4])
X_test_filenames = X_test.map(lambda x: byte_data_directory+x+'.bytes')
X_test_filenames = X_test_filenames.collect()
dat_test = sc.wholeTextFiles(",".join(X_test_filenames))
"""

"""

dat_test = dat_test.mapValues(lambda x: x.split())
dat_test = dat_test.mapValues(lambda x: list(filter(lambda y: len(y) == 2, x)))
test_df = dat_test.toDF(['filename', 'words'])
preds = model.transform(test_df)
preds.show()
"""
