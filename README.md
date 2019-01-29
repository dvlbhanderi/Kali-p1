# Kali-p1
Malware classification project repo for team Kali

This repo consist of a large scale classifier to classify the documents as being under one of the following nine categories :-
1.) Rammit
2.) Lollipop
3.) Kelihos_ver3
4.) Vundo
5.) Simda
6.) Tracur
7.) Kelihos_ver1
8.) Obfuscator.ACY
9.) Gatak

# Prerequisites
// TBD

# Dependcies
//TBD

# Built With
Python 3.6
Apache Spark
PySpark - Python API for Apache Spark

# Data
Data from the Microsoft Malware Classification Challenge is used for this project.
  ## Data Location
  https://storage.googleapis.com/uga-dsp/project1/data/bytes/<file>
  OR
  gs://uga-dsp/project1/data/bytes/<file>
  
  ## File Structure
  Each file looks as follows :
  
  00401060 53 8F 48 as 00 87 ad ds
  00401070 43 4F 58 Fs 40 47 Fd Gs
  00401060 63 6F 68 Gs 60 67 Gd Gs
  00401060 13 1F 18 Ws 10 17 Wd Ws
  00401060 23 2F 28 Ts 20 27 Td Ts
  
  ## File Interpretation
  The first hexadecimal token in each line just indicates the line pointer, hence no role in classification and is ignored.
  All the other hexadecimal pairs are the code of the malware instance and are used for prediction. 

# Contributing
//TBD

# Authors
See the contributors file for details.

# Licencse
See the License.md file for details.


