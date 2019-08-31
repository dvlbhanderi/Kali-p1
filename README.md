# Team Kali: Microsoft Malware Classification

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
      
      00401060 53 8F 48 as 00 87 ad ds 
      00401070 43 4F 58 Fs 40 47 Fd Gs
      00401060 63 6F 68 Gs 60 67 Gd Gs
      00401060 13 1F 18 Ws 10 17 Wd Ws
      00401060 23 2F 28 Ts 20 27 Td Ts
      
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
    
   ### logistic_regression.py
   
The pipeline consists of four stages: indexer, regextokenizer, hashingTF, and lr

CrossValidator begins by splitting the dataset into a set of folds which are used as separate training and test datasets. E.g., with k=3 folds, CrossValidator will generate 3 (training, test) dataset pairs, each of which uses 2/3 of the data for training and 1/3 for testing. To evaluate a particular ParamMap, CrossValidator computes the average evaluation metric for the 3 Models produced by fitting the Estimator on the 3 different (training, test) dataset pairs.

cross-validation over a grid of parameters is expensive. E.g., in the example below, the parameter grid has 3 values for hashingTF.numFeatures and 2 values for lr.regParam, and CrossValidator uses 3 folds. This multiplies out to (3×2)×3=18 different models being trained. 
   
   The Script can be run as spark-submit logistic_regression.py [args]
   The list of arguments to be passed is the same as  give above in the spark_NB.py section.
   Eg:

     spark-submit logistic_regression.py /path/to/list/of/train-file-names /path/to/list/of/train-labels /path/to/list/of/test-file-names /path/to/list/of/test-labels /path/to/bytefiles-directory

    
   ### random_forest.py

This script should be run as spark-submit random_forest.py[args]

To accept the arguments the script uses argparse library of python.

All the arguments should be passed with the argument label as specified in the script. All arguments are not compulsory,
and can be specified as per requirements.

Following arguments can be used in the script :-

1. --mode : can be train or test depending on the phase of execution.
2. --filename_path : path of the directory where hashcodes are stored.
3. --filelabel_path : path of the directory where the labels of each file are stored.
4. --data_path : path of the directory where the actual data is stored.
5. --save_path : path where user wants to save the data during training phase or testing phase.
6. --model_path : path from where the random forest model and countvector model created during training phase  has to be loaded.
7. --exec_mem : memory space that user wants to allocate for the executors.
8. --driver_mem : memory space that user wants to allocate for the driver.
9. --result_mem : memory space that user wants to allocate for the result.

Each of these labels could be passed with the desired values while running the script.

Eg Training:-
```
spark-submit random_forest.py --mode train --filename_path /path/to/filename --filelabel_path /path/to/labels --data_path /path/to/data --save_path /path/to/store --exec_mem '20G' --driver_mem '40G' --result_mem '12G'
```
Eg Testing :-
```
spark-submit random_forest.py --mode test --filename_path /path/to/filename --data_path /path/to/data --save_path /path/to/store  --model_path /path/to/load/model --exec_mem '20G' --driver_mem '40G' --result_mem '12G'
```

For more information on random forest refer 'References' section. 
    
# Execution on the Google Cloud Platform
1) Set up a [project](https://cloud.google.com/dataproc/docs/guides/setup-project)
2) Make a storage bucket on the [Storage](https://cloud.google.com/storage/docs/creating-buckets) service of the GCP and        upload the .py file on it.
3) Apache Spark models are supported on Dataproc. Enable Compute Engine and Dataproc
4) Now you can create a [Cluster](https://cloud.google.com/dataproc/docs/guides/create-cluster). Issue #29 of this repo        gives a snapshot of the setup of the cluster.
5) Create a Job and give the the Google storage path of the .py file (created in step 2) in the main file section of the        Job. [Submit](https://cloud.google.com/dataproc/docs/guides/submit-job) the job on the created cluster.


# References
* https://en.wikipedia.org/wiki/Random_forest
* https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd
* https://www.youtube.com/watch?v=D_2LkhMJcfY
* https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
* https://en.wikipedia.org/wiki/Random_forest
* https://en.wikipedia.org/wiki/Naive_Bayes_classifier
* https://www.statisticssolutions.com/what-is-logistic-regression/
* https://en.wikipedia.org/wiki/Apache_Spark
* https://en.wikipedia.org/wiki/Python_(programming_language)
* 


# Authors
See the [Contributors](https://github.com/dsp-uga/Kali-p1/blob/master/CONTRIBUTORS.md) file for details.

# Licencse
See the [License](https://github.com/dsp-uga/Kali-p1/blob/master/LICENSE) file for details.



