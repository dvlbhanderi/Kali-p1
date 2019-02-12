#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pyspark
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import CountVectorizer,CountVectorizerModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import RegressionEvaluator,MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import argparse


# In[2]:

def configure_spark(exec_mem, driver_mem, result_mem):
    '''
    This function configures spark. It accepts as input the memory to be allocated
    to each executor, memory to be allocated to the driver and memory to be allocated
    for the output.

    Argument 1(String) : Memory to be allocated to the executors
    Argument 2(String) : Memory to be allocated to the driver
    Argument 3(String) : Max memory to be allocated for the result

    '''
    conf = pyspark.SparkConf().setAppName('Malware Classification')
    conf = (conf.setMaster('local[*]')
       .set('spark.executor.memory', exec_mem)
       .set('spark.driver.memory', driver_mem)
       .set('spark.driver.maxResultSize', result_mem))
    sc = pyspark.SparkContext(conf=conf)
    Spark = SparkSession(sc)
    return sc, Spark



# In[3]:


def readFile(path):
    '''
    This function reads the files from the given path and return an rdd containing
    file data.

    Arg1: path of the directory of the files.

    '''
    return sc.textFile(path,minPartitions = 32)


# In[4]:


def readWholeFile(path):
    '''
    This function reads the files along with filenames from the given path
    and return an rdd containing filename and its data.

    Arg1: Path of the directory of the files.

    '''
    return sc.wholeTextFiles(path, minPartitions = 32)


# In[5]:


def readTrainingFiles(filename_path, filelabel_path, data_path):
    '''
    This function reads the name of the training files, their labels
    and their data given each of these paths and returns an rdd containing
    the data and an rdd containing the labels.

    Arg1 : Path of the file storing the name of the files.
    Arg2 : Path of the file storing the labels of these files.
    Arg3 : Path where the data is stored.

    '''
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
    label_map = dict(zip(x_filenames,y_train.collect()))
    #converting it to rdd
    labelfile_rdd = sc.parallelize(label_map)


    return dat_train, labelfile_rdd



# In[7]:


def preprocessing_trainingfiles(labelfile_rdd, dat_rdd):
    '''
    This function preprocess the training files and returns and rdd containing
    the filenames,data and their labels.
    Arg1 : rdd containing the labels and filenames
    Arg2 : rdd containing the data and filenames.
    '''
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
    '''
    This function preprocess the testing files and returns the
    preprocessed rdd.
    Arg1(rdd) : rdd containing the testing data.
    '''
    #Removed the filepointers from the files
    testing_data = testing_data.map(lambda x : (x[0],x[1].split()[1:]))

    #shortening the full filepath to just the filename for the data
    testing_data = testing_data.map(lambda x : (x[0].split('/')[-1],x[1]))

    return testing_data


# In[9]:


def rddToDf_training(dat_rdd):
    '''
    This function converts rdd of the training files to the data frame and returns the dataframe.
    Arg1(rdd) : rdd to be converted to dataframe.
    '''
    #converting the rdd to dataframe with labels
    print('*********** inside to convert into dataframe *********************')
    print('*********** inside to convert into dataframe *********************')
    print('*********** inside to convert into dataframe *********************')

    final_df = dat_rdd.map(lambda line : Row(data = line[1][1],label = line[1][0], filename = line[0])).toDF()
    return final_df


# In[10]:


def rddToDf_testing(dat_rdd):

    '''
    This function converts rdd of the testing files to the data frame and returns the dataframe.
    Arg1(rdd) : rdd to be converted to dataframe.
    '''
    #converting the rdd to dataframe with labels
    final_df = dat_rdd.map(lambda line : Row(data = line[1], filename = line[0])).toDF()
    #final_df = dat_rdd.map(lambda line : Row(data = line[1][1], label = line[1][0], filename = line[0])).toDF()

    return final_df


# In[11]:


def getCountVector(final_df):
    '''
    This function accepts as input a dataframe with a column named 'data' containing
    each document as a row. This will be converted to a countvector and the ouput column
    will be named 'indexedFeatures'. It returns the countvector model and the original dataframe
    with additional column 'indexedFeatures'.

    Arg1 : dataframe to compute the count vector

    '''
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


def typeCastColumn(countVector_df):
    '''
    This function type casts column of a dataframe to the specified data type,
    and returns the modified dataframe.
    Arg1 : dataframe whose column has to be typecasted.
    Arg2 : name of the column which has to be typecasted.
    Arg3 : Data type to which it has to be type casted.
    '''

    print('************ insdie type casting **************')
    print('************ insdie type casting **************')
    print('************ insdie type casting **************')

    #typecasting the labels to the integer datatype
    final_df = countVector_df.withColumn('label', countVector_df['label'].cast('int'))

    print('************* returning from type casting ************')
    print('************* returning from type casting ************')
    print('************* returning from type casting ************')

    return final_df


# In[13]:


def train_random_forest(final_df):
    '''
    This function accepts a dataframe as an input and train the machine using
    this data on randomforest algorithm to generate a model and returns the model.

    Arg1 : dataframe on which model has to be trained.
    '''

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
    '''
    This functoin accepts as input the model previously trained and the data on which
    prediction has to be made and returns the predictions.

    Arg1 : Model obtained from training.
    Arg2 : Data on which predictions has to be made.
    '''
    predictions = rfModel.transform(data)
    return predictions


# In[15]:


def evaluate_accuracy(predictions):
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    return (evaluator.evaluate(predictions)*100)



# In[ ]:


if __name__ == '__main__':


    arguments = argparse.ArgumentParser(description='inputs')
    arguments.add_argument('--mode',type=str,help='train or test')
    arguments.add_argument('--filename_path',type=str,help='path of the hashcodes')
    arguments.add_argument('--filelabel_path',type=str,help='path of the labels of each document')
    arguments.add_argument('--data_path',type=str,help='path of the actual data')
    arguments.add_argument('--save_path',type=str,help='path to save data or model')
    arguments.add_argument('--model_path', type=str,help='path to load the models from, during testing')
    arguments.add_argument('--exec_mem', type=str,help='memory to be allocated to executors')
    arguments.add_argument('--driver_mem', type=str,help='memory to be allocated to driver')
    arguments.add_argument('--result_mem', type=str,help='memory to be allocated for the result')
    args = arguments.parse_args()

    print('************************* INSIDE MAIN ****************************')
    print('************************* INSIDE MAIN ****************************')
    print('************************* INSIDE MAIN ****************************')


    #configuring spark
    sc, Spark = configure_spark('Malware Classification',args.exec_mem,args.driver_mem,args.result_mem)

    print('********************** after configuration **************************')
    print('********************** after configuration **************************')
    print('********************** after configuration **************************')

    if args.mode=="train":
        #reading training data
        train_data, labelfile_rdd = readTrainingFiles(args.filename_path,args.filelabel_path,args.data_path)

        print('**************** after reading training files **********************')
        print('**************** after reading training files **********************')
        print('**************** after reading training files **********************')

        #preprocessing the training data
        train_data = preprocessing_trainingfiles(labelfile_rdd, train_data)

        print('************* after preprocessing *****************')
        print('************* after preprocessing *****************')
        print('************* after preprocessing *****************')

        #converting the training data rdd to df
        train_data = rddToDf_training(train_data)

        print('***************** after converting rdd to dataframe ***************')
        print('***************** after converting rdd to dataframe ***************')
        print('***************** after converting rdd to dataframe ***************')

        #converting data into count vectors for training
        train_data,cv = getCountVector(train_data)

        print('***************** after getting the count vector ***************')
        print('***************** after getting the count vector ***************')
        print('***************** after getting the count vector ***************')

        #saving the training data and countvectorizer model
        train_data.write.parquet( args.save_path + '/traindata.parquet')
        cv.save( args.save_path + '/countvector_model')

        #type casting field 'label' from string to int
        train_data = typeCastColumn(train_data,'label','int')

        print('************** after typecasting *************')
        print('************** after typecasting *************')
        print('************** after typecasting *************')

        #training the model on random forest
        rfModel = train_random_forest(train_data)

        #saving the model as parquet file
        rfModel.save(args.save_path + '/rfmodel')


        print('********************* after model training *****************')
        print('********************* after model training *****************')
        print('********************* Training Completed *******************')

    elif args.mode=="test":
        #reading the testing files
        testing_data = readTestingFiles(args.filename_path, args.data_path)

        print('********************* after reading testing files *****************')
        print('********************* after reading testing files *****************')
        print('********************* after reading testing files *****************')

        #preprocessing testing files
        testing_data = preprocessing_testingfiles(testing_data)

        print('********************* after preprocessing testing files *****************')
        print('********************* after preprocessing testing files *****************')
        print('********************* after preprocessing testing files *****************')

        #converting the testing rdd to df
        testing_data = rddToDf_testing(testing_data)

        print('********************* after converting to df *****************')
        print('********************* after converting to df *****************')
        print('********************* after converting to df *****************')

        #reading the saved countvector model
        cv = CountVectorizerModel.load(args.model_path + '/countvector_model')
        #transforming test data to count vector
        testing_data = cv.transform(testing_data)
        #saving the transformed data as parquet file
        testing_data.write.parquet(args.model_path + '/testingdata.parquet')

        print('********************* after cv transformation *****************')
        print('********************* after cv transformation *****************')
        print('********************* after cv transformation  *****************')

        #reading the saved random forest model
        rfModel = RandomForestClassificationModel.load(args.model_path + '/rfmodel' )
        #getting the predictions
        predictions = predict(rfModel, testing_data)

        #saving the predictions as parquet file
        predictions.write.parquet(args.model_path + '/predictions.parquet')

        print('********************* after predicitons  *****************')
        print('********************* after predicitons  *****************')
        print('********************* Done  *****************')

    else:
        print("Enter correct mode (train or test)")
