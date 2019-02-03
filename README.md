# Kali-p1

This repo consist of a large scale classifier to classify the documents as being under one of the following nine categories :-
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

Make sure you have all the prerequisites installed before proceeding. Skipping or missing certain dependencies will give you some errors that will waste a considerable amount of your time. Especially while installing Apache Spark, keep in mind that there are some versions of Java that are not yet supported.

### Prerequisites

List of requirements and links to install them

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/) - Python Environment virtualization.
- [Apache Spark](https://spark.apache.org/downloads.html)
- [Pyspark setup for Windows](https://medium.com/@GalarnykMichael/install-spark-on-windows-pyspark-4498a5d8d66c) 
- [Pyspark setup for Ubuntu](https://medium.com/@GalarnykMichael/install-spark-on-ubuntu-pyspark-231c45677de0)
- [Pyspark setup for MacOS](https://medium.com/@GalarnykMichael/install-spark-on-mac-pyspark-453f395f240b)

# Dependcies
//TBD


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
  The first hexadecimal token in each line just indicates the line pointer, hence no role in classification and is ignored.
  All the other hexadecimal pairs are the code of the malware instance and are used for prediction. 

# Contributing
//TBD

# Authors
See the [Contributors](https://github.com/dsp-uga/Kali-p1/blob/master/CONTRIBUTORS.md) file for details.

# Licencse
See the [License](https://github.com/dsp-uga/Kali-p1/blob/master/LICENSE) file for details.


