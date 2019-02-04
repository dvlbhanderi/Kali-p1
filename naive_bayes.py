import sys
from math import log
from pyspark import SparkContext
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from operator import add
sc = SparkContext.getOrCreate()



if __name__ == '__main__':
    """
    This is the driver of the malware_classifier, it should take in
    two files for training with filenames (argv[1]) to pull in from a
    directory of malware byte files, and the respective labels
    (argv[2]) for each. This should then take a directory of byte
    files (argv[3]) to read in based on the filenames provided.
    This then repeats the process to read in test data defined by
    the testing filenames (argv[4])
    Once the training data is fully read in and distributed, the
    pipeline for training will call each of the functions defined
    above to train the classifier and apply it to test input
    """

    # read in the training filenames and labels
    X_train = sc.textFile(sys.argv[1])
    y_train = sc.textFile(sys.argv[2])

    # read directory of byte files
    byte_data_directory = sys.argv[3]

    # form full filename from directory and names and loas into list
    X_filenames = X_train.map(lambda x: byte_data_directory+x+'.bytes')
    X_filenames = X_filenames.collect()

    # load pairRDD of text to preproc and train on
    dat_train = sc.wholeTextFiles(",".join(X_filenames))

    # create constant time map of filename to label
    label_map = dict(zip(X_filenames,y_train.collect()))

    # read in the testing data using same process as above
    X_test = sc.textFile(sys.argv[4])
    X_test_filenames = X_test.map(lambda x: byte_data_directory+x+'.bytes')
    X_test_filenames = X_test_filenames.collect()
dat_test = sc.wholeTextFiles(",".join(X_test_filenames))
