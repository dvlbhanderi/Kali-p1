# Kali-p1

This repo consists of several efforts to tackle document classification

Our work was focused on the dataset from the [Microsoft Malware Classification Challenge](https://www.kaggle.com/c/malware-classification/). The problem is to classify documents as being under one of the following nine malware categories :-
* 1.) Rammit
* 2.) Lollipop
* 3.) Kelihos_ver3
* 4.) Vundo
* 5.) Simda
* 6.) Tracur
* 7.) Kelihos_ver1
* 8.) Obfuscator.ACY
* 9.) Gatak

## Approach
We explored a few different approaches. Initially we begain implementing our own Naive Bayes classifier. However due to time constraints and memory inefficiencies we transitioned to building pipelines using preimplemented modules in the Apache Spark Pyspark API.

#### Naive Bayes
Our first pipeline focused on using a Naive Bayes classifier. It can be found in src/spark_NB.py. It uses a regex tokenizer to remove line pointers from the byte files and tokenize the bytewords. The resulting tokens are passed into a n-gram featurizer, which coerces the tokens into bigrams. The bigrams are then counted using Spark's hashing term frequency method [here](https://spark.apache.org/docs/2.2.0/api/python/pyspark.mllib.html#pyspark.mllib.feature.HashingTF). These hashed word counts are used as the features in a Naive Bayes model with a simple additive smoothing value of one. With this pipeline we achieved an accuracy of 80%

## Getting Started

Make sure you have all the prerequisites installed before proceeding. Skipping or missing certain dependencies will give you some errors that will waste a considerable amount of your time. Especially while installing Apache Spark, keep in mind that there are some versions of Java that are not yet supported. Also add winutils.exe file separately while installing Apache Spark on Windows.

### Prerequisites

List of requirements and links to install them:

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/) - Python Environment virtualization.
- [Apache Spark](https://spark.apache.org/downloads.html)
- [Pyspark setup for Windows](https://medium.com/@GalarnykMichael/install-spark-on-windows-pyspark-4498a5d8d66c) 
- [Pyspark setup for Ubuntu](https://medium.com/@GalarnykMichael/install-spark-on-ubuntu-pyspark-231c45677de0)
- [Pyspark setup for MacOS](https://medium.com/@GalarnykMichael/install-spark-on-mac-pyspark-453f395f240b)
- [Google Cloud Platform or similar service](https://cloud.google.com/docs/)

# Data
Data from the Microsoft Malware Classification Challenge is used for this project.
  ## Data Location
* https://storage.googleapis.com/uga-dsp/project1/data/bytes/<file>
OR
* gs://uga-dsp/project1/data/bytes/<file>

  ## File Structure
      Each file looks as follows :
      '''
      00401060 53 8F 48 as 00 87 ad ds 
      00401070 43 4F 58 Fs 40 47 Fd Gs
      00401060 63 6F 68 Gs 60 67 Gd Gs
      00401060 13 1F 18 Ws 10 17 Wd Ws
      00401060 23 2F 28 Ts 20 27 Td Ts
      '''
  ## File Interpretation
The first hexadecimal token in each line just indicates the line pointer, hence plays no role in classification and is       ignored.
All the other hexadecimal pairs are the code of the malware instance and are used for prediction. 

# Included Scripts
All source code is included in the src directory of the repository

  ### spark-tests.py: (development discontinued due to time constraint)
This file can be run using '$ python spark-tests.py' and will boot up a local spark session and test each of the functions in malware_classifier.py
  ### malware_classifier.py: (development discontinued due to time constraint)
This script must be submitted to a spark session as a job using '$ spark-submit malware_classifier [args]'
    
The first argument should be the location of a text file containing names of byte files to train the model
    
The second argument should be the location of a text file containing training labels
    
The third argument should be the directory of all of the byte files
    
The fourth argument should be the location of a text file containign names of byte files to classify
  ### spark_NB.py:
This script creates a pipeline around the spark implementation of Naive Bayes using a regex tokenizer to split the words and creates a list of bigrams per each document. A Term frequency feature list is created using spark's hashing TF. This featurization is passed to the Naive Bayes model. Once the model is trained, it will fit the testing dataset and print the predictions to standard out. These can be copied and pasted into another buffer or piped into a text file as desired.
    
The Script can be run as spark-submit spark_NB.py [args]
    
The first argument should be the location of a text file containing names of byte files to train the model
    
The second argument should be the location of a test file containing training labels
    
the third argument should be the location of a text file containg names of files to classify
    
the fourth argument should be the location of testing labels ('None' should be provided if there are no testing files)
    
The fifth argument should be the directory of the byte files to be used
    
   ### random_forest.py
This script implements random forest algorithm on the given dataset to generate predictions.

It is divided into following steps :-
1. Configuring the spark session.
2. Reading the training files.
3. Preprocessing the training files.
4. Converting the training files rdd to dataframe for further operation.
5. Saving the dataframe formed as parquet file on google cloud.  
6. Getting the count vector of the training data.
7. Saving the transformed training data and count vectorizer as parquet file on google cloud.
8. Type casting the column 'label' in the training data to 'int' type before training the model.
9. Training the random forest model.
10. Saving the obtained model as parquet file on google cloud.
11. Reading the testing files.
12. Preprocessing the testing files.
13. Converting the testing file rdd to dataframe.
14. Transforming the testing file rdd using countvectorizer.
15. Saving the transformed testing file as a parquet file on google cloud.
16. Getting the predictions.
17. Saving the predictions on google cloud.

Steps #5, #7, #10, #15 could be skipped while running this script by commenting out the corresponding lines in the code.
But is strongly recommended, as saving the files can eliminate the need to re run the whole script in case any error is     encountered during execution of the script. We can always start from the point of failure if our data is saved.

Instruction to load the parquet file :


For information on random forest algorithm, you can refer following pages :-
* https://en.wikipedia.org/wiki/Random_forest
* https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd
* https://www.youtube.com/watch?v=D_2LkhMJcfY
    
    
    
# Execution on the Google Cloud Platform
1) Set up a [project](https://cloud.google.com/dataproc/docs/guides/setup-project)
2) Make a storage bucket on the [Storage](https://cloud.google.com/storage/docs/creating-buckets) service of the GCP and        upload the .py file on it.
3) Apache Spark models are supported on Dataproc. Enable Compute Engine and Dataproc
4) Now you can create a [Cluster](https://cloud.google.com/dataproc/docs/guides/create-cluster). Issue #29 of this repo        gives a snapshot of the setup of the cluster.
5) Create a Job and give the the Google storage path of the .py file (created in step 2) in the main file section of the        Job. [Submit](https://cloud.google.com/dataproc/docs/guides/submit-job) the job on the created cluster.

# Authors
See the [Contributors](https://github.com/dsp-uga/Kali-p1/blob/master/CONTRIBUTORS.md) file for details.

# Licencse
See the [License](https://github.com/dsp-uga/Kali-p1/blob/master/LICENSE) file for details.


