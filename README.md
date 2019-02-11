# Kali-p1

This repo consists of a large scale classifier to classify the documents as being under one of the following nine malware categories :-
* 1.) Rammit
* 2.) Lollipop
* 3.) Kelihos_ver3
* 4.) Vundo
* 5.) Simda
* 6.) Tracur
* 7.) Kelihos_ver1
* 8.) Obfuscator.ACY
* 9.) Gatak


## Getting Started

Make sure you have all the prerequisites installed before proceeding. Skipping or missing certain dependencies will give you some errors that will waste a considerable amount of your time. Especially while installing Apache Spark, keep in mind that there are some versions of Java that are not yet supported. Also add winutils.exe file separately while installing Apache Spark on Windows.

### Prerequisites

List of requirements and links to install them

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
  https://storage.googleapis.com/uga-dsp/project1/data/bytes/<file>
  OR
  gs://uga-dsp/project1/data/bytes/<file>
  
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
  The first hexadecimal token in each line just indicates the line pointer, hence plays no role in classification and is ignored.
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
    This script creates a pipeline around the spark implementation of Naive Bayes using a regex tokenizer to split the words and creates a list of bigrams per each document. A Term frequency feature list is created using spark's hashing TF. This featurization is passed to the Naive Bayes model.
    
    The Script can be run as spark-submit spark_NB.py [args]
    
    The first argument should be the location of a text file containing names of byte files to train the model
    
    The second argument should be the location of a test file containing training labels
    
    the third argument should be the location of a text file containg names of files to classify
    
    the fourth argument should be the location of testing labels ('None' should be provided if there are no testing files)
    
    The fifth argument should be the directory of the byte files to be used
    
# Execution on the Google Cloud Platform
1) Set up a [project](https://cloud.google.com/dataproc/docs/guides/setup-project)
2) Make a storage bucket on the [Storage](https://cloud.google.com/storage/docs/creating-buckets) service of the GCP and upload the .py file on it.
3) Apache Spark models are supported on Dataproc. Enable Compute Engine and Dataproc
4) Now you can create a [Cluster](https://cloud.google.com/dataproc/docs/guides/create-cluster). Issue #29 of this repo gives a snapshot of the setup of the cluster.
5) Create a Job and give the the Google storage path of the .py file (created in step 2) in the main file section of the Job. [Submit](https://cloud.google.com/dataproc/docs/guides/submit-job) the job on the created cluster.

# Authors
See the [Contributors](https://github.com/dsp-uga/Kali-p1/blob/master/CONTRIBUTORS.md) file for details.

# Licencse
See the [License](https://github.com/dsp-uga/Kali-p1/blob/master/LICENSE) file for details.


