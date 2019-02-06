#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pyspark
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import  CountVectorizer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator,MulticlassClassificationEvaluator
from pyspark.sql import SparkSession


# In[2]:


def configure_spark():
    conf = pyspark.SparkConf().setAppName("Malware_Classification")
    conf = (conf.setMaster('local[*]')
       .set('spark.executor.memory', '20G')
       .set('spark.driver.memory', '40G')
       .set('spark.driver.maxResultSize', '12G'))
    sc = pyspark.SparkContext(conf=conf)
    Spark = SparkSession(sc)
    return sc, Spark



# In[3]:


def readFile(path):
    return sc.textFile(path)


# In[4]:


def readWholeFile(path):
    return sc.wholeTextFiles(path)


# In[5]:


def readTrainingFiles(filename_path, filelabel_path, data_path):
    # reading the training files
    x_train = readFile(filename_path)
    y_train = readFile(filelabel_path)

    #storing the path of data
    byte_data_directory = data_path

    #appending hashcode of the files to the datapath
    x_filenames = x_train.map(lambda x: byte_data_directory+x+'.bytes')
    x_filenames = x_filenames.collect()

    #reading the data
    dat_train = readWholeFile(",".join(x_filenames))

    #making the list of filename and its labels
    label_map = list(zip(x_filenames,y_train.collect()))
    #converting it to rdd
    labelfile_rdd = sc.parallelize(label_map)


    return dat_train, labelfile_rdd






# In[6]:


def readTestingFiles(filename_path, data_path):
    # reading the training files
    x_test = readFile(filename_path)

    #storing the path of data
    byte_data_directory = data_path

    #appending hashcode of the files to the datapath
    x_filenames = x_test.map(lambda x: byte_data_directory+x+'.bytes')
    x_filenames = x_filenames.collect()

    #reading the data
    dat_test = readWholeFile(",".join(x_filenames))

    return dat_test


# In[7]:


def preprocessing_trainingfiles(labelfile_rdd, dat_rdd):
    #shortening the full filepath to just the filename for the label
    labelfile_rdd = labelfile_rdd.map(lambda x : (x[0].split('/')[-1],x[1]))

    #Removed the filepointers from the files
    dat_rdd = dat_rdd.map(lambda x : (x[0],x[1].split()[1:]))

    #shortening the full filepath to just the filename for the data
    dat_rdd = dat_rdd.map(lambda x : (x[0].split('/')[-1],x[1]))

    #joined the rdd containing label and filename to the rdd with data and filename
    dat_rdd = labelfile_rdd.join(dat_rdd)

    return dat_rdd





# In[8]:


def preprocessing_testingfiles(testing_data):
    #Removed the filepointers from the files
    testing_data = testing_data.map(lambda x : (x[0],x[1].split()[1:]))

    #shortening the full filepath to just the filename for the data
    testing_data = testing_data.map(lambda x : (x[0].split('/')[-1],x[1]))

    return testing_data


# In[9]:


def rddToDf_training(dat_rdd):
    #converting the rdd to dataframe with labels
    print('*********** inside to convert into dataframe *********************')
    print('*********** inside to convert into dataframe *********************')
    print('*********** inside to convert into dataframe *********************')

    final_df = dat_rdd.map(lambda line : Row(data = line[1][1],label = line[1][0], filename = line[0])).toDF()
    return final_df


# In[10]:


def rddToDf_testing(dat_rdd):
    #converting the rdd to dataframe with labels
    final_df = dat_rdd.map(lambda line : Row(data = line[1], filename = line[0])).toDF()
    return final_df


# In[11]:


def getCountVector(final_df):
    #getting the countvector
    print('************* inside the count vector ****************')
    print('************* inside the count vector ****************')
    print('************* inside the count vector ****************')

    cv = CountVectorizer(inputCol = "data", outputCol = "indexedFeatures").fit(final_df)
    countVector_df = cv.transform(final_df)

    print('************* returning the count vector ****************')
    print('************* returning the count vector ****************')
    print('************* returning the count vector ****************')


    return countVector_df,cv


# In[12]:


def typeCastColumn(countVector_df,columnName,datatype):
    '''this function type casts column of a dataframe to the specified data type.
    It accepts three parameters as input, 1. dataframe 2. name of the column to be type
    casted 3. data type to which it should be type casted'''

    print('************ insdie type casting **************')
    print('************ insdie type casting **************')
    print('************ insdie type casting **************')

    #typecasting the labels to the integer datatype
    final_df = countVector_df.withColumn(columnName, countVector_df[columnName].cast(datatype))

    print('************* returning from type casting ************')
    print('************* returning from type casting ************')
    print('************* returning from type casting ************')

    return final_df


# In[13]:


def train_random_forest(final_df):
    '''This function accepts a dataframe as an input and train the machine using
    this data through randomforest'''

    print('********* inside training random forest **************')
    print('********* inside training random forest ************')
    print('********* inside training random forest ************')


    rf = RandomForestClassifier(labelCol = "label", featuresCol = "indexedFeatures", numTrees = 20, maxDepth = 8, maxBins = 32)
    #training the model
    rfModel = rf.fit(final_df)

    print('********** returning from random forest *************')
    print('********** returning from random forest *************')
    print('********** returning from random forest *************')


    return rfModel


# In[14]:


def predict(rfModel, data):
    predictions = rfModel.transform(data)
    return predictions


# In[15]:


def evaluate_accuracy(predictions):
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    return (evaluator.evaluate(predictions)*100)



# In[ ]:


if __name__ == '__main__':

    print('************************* INSIDE MAIN ****************************')
    print('************************* INSIDE MAIN ****************************')
    print('************************* INSIDE MAIN ****************************')


    sc, Spark = configure_spark()

    print('********************** after configuration **************************')

    print('********************** after configuration **************************')

    print('********************** after configuration **************************')

    train_data, labelfile_rdd = readTrainingFiles('gs://uga-dsp/project1/files/X_small_train.txt','gs://uga-dsp/project1/files/y_small_train.txt','gs://uga-dsp/project1/data/bytes/')

    print('**************** after reading training files **********************')
    print('**************** after reading training files **********************')
    print('**************** after reading training files **********************')


    print('************** before preprocessing *****************')
    print('************** before preprocessing *****************')
    print('************** before preprocessing *****************')

    train_data = preprocessing_trainingfiles(labelfile_rdd, train_data)
    print('************* after preprocessing *****************')
    print('************* after preprocessing *****************')
    print('************* after preprocessing *****************')


    print('***************** before converting rdd to dataframe **************')
    print('***************** before converting rdd to dataframe **************')
    print('***************** before converting rdd to dataframe ***************')
    train_data = rddToDf_training(train_data)

    print('***************** after converting rdd to dataframe ***************')
    print('***************** after converting rdd to dataframe ***************')
    print('***************** after converting rdd to dataframe ***************')


    print('***************** before getting the count vector ***************')
    print('***************** before getting the count vector ***************')
    print('***************** before getting the count vector ***************')


    train_data,cv = getCountVector(train_data)

    print('***************** after getting the count vector ***************')
    print('***************** after getting the count vector ***************')
    print('***************** after getting the count vector ***************')

    print('*************** before writing the count vector to the bucket *************')
    print('*************** before writing the count vector to the bucket *************')
    print('*************** before writing the count vector to the bucket *************')


    train_data.write.parquet('gs://project1_dsp_priyank/data/traindata_cv.parquet')
    cv.save('gs://project1_dsp_priyank/data/countvectorizer')

    print('********************* after count vectorizer *****************')
    print('********************* after count vectorizer *****************')
    print('********************* after count vectorizer *****************')
    train_data = typeCastColumn(train_data,'label','int')

    print('************** before training the model *************')
    print('************** before training the model *************')
    print('************** before training the model *************')

    rfModel = train_random_forest(train_data)

    print('********************* after model training *****************')
    print('********************* after model training *****************')
    print('********************* after model training *****************')

    testing_data = readTestingFiles('gs://uga-dsp/project1/files/X_small_test.txt', 'gs://uga-dsp/project1/data/bytes/')

    testing_data = preprocessing_testingfiles(testing_data)

    testing_data = rddToDf_testing(testing_data)

    testing_data = cv.transform(testing_data)
    testing_data.write.parquet('gs://project1_dsp_priyank/data/testing_data.parquet')

    print('********************* after testing  *****************')
    print('********************* after testing  *****************')
    print('********************* after testing  *****************')


    predictions = predict(rfModel, testing_data)
    predictions.write.parquet('gs://project1_dsp_priyank/data/predictions.parquet')


    print('********************* after predicitons  *****************')
    print('********************* after predicitons  *****************')
    print('********************* after predicitons  *****************')


    accuracy = evaluate_accuracy(predictions)
    print('The accuracy for the code by Arch Man and Priyank is :' , accuracy )
