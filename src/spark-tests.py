import unittest
import logging
from operator import add
from pyspark.sql import SparkSession

import malware_classifier


class PySparkTest(unittest.TestCase):

    @classmethod
    def suppress_py4j_logging(cls):
            logger = logging.getLogger('py4j')
            logger.setLevel(logging.WARN)

    @classmethod
    def create_testing_pyspark_session(cls):
            return (SparkSession.builder
                                .master('local[2]')
                                .appName('local-testing-pyspark-context')
                                .enableHiveSupport()
                                .getOrCreate())

    @classmethod
    def setUpClass(cls):
        cls.suppress_py4j_logging()
        cls.spark = cls.create_testing_pyspark_session()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


class SimpleTest(PySparkTest):

    def test_tokenization_and_preproc(self):
        """
        This function will test the tokenize_and_preproc unit of the driver
        module. That unit should recieve a pairRDD of file names and file
        bodies split the body on whitespace to tokenize and filter out line
        pointer.
        """

        dat = [('filename1',
                '00401060 53 8F 48 00 A9 88 40 00 04 4E 00 00 F9 31 4F 00'),
               ('filename2',
                '00401070 1D 99 02 47 D5 4F 00 00 03 05 B5 42 CE 88 65 43')]
        test_rdd = self.spark.sparkContext.parallelize(dat, 2)
        results = malware_classifier.tokenize_and_preproc(test_rdd).collect()
        expected_results = [('filename1', '53'), ('filename1', '8F'),
                            ('filename1', '48'), ('filename1', '00'),
                            ('filename1', 'A9'), ('filename1', '88'),
                            ('filename1', '40'), ('filename1', '00'),
                            ('filename1', '04'), ('filename1', '4E'),
                            ('filename1', '00'), ('filename1', '00'),
                            ('filename1', 'F9'), ('filename1', '31'),
                            ('filename1', '4F'), ('filename1', '00'),
                            ('filename2', '1D'), ('filename2', '99'),
                            ('filename2', '02'), ('filename2', '47'),
                            ('filename2', 'D5'), ('filename2', '4F'),
                            ('filename2', '00'), ('filename2', '00'),
                            ('filename2', '03'), ('filename2', '05'),
                            ('filename2', 'B5'), ('filename2', '42'),
                            ('filename2', 'CE'), ('filename2', '88'),
                            ('filename2', '65'), ('filename2', '43')]
        self.assertEqual(set(results), set(expected_results))

    def test_prior_calculations(self):
        """
        This function tests the calculation the priors for each class by counting
        the occurances of each class type in a y_train set and dividing by the
        number of documents which exist
        """
        dat = ['6', '3', '1', '7', '9', '1', '6', '3', '3', '7', '2', '1', '6',
               '8', '2', '3', '2', '5', '3', '3', '1', '4', '3', '2', '8', '8',
               '8', '2', '4', '3', '6', '5', '2', '9', '7', '6', '5', '4', '3',
               '2', '2', '2', '5', '2', '1', '7', '3', '9', '8', '3']
        test_rdd = self.spark.sparkContext.parallelize(dat, 2)
        results = malware_classifier.calculate_priors(test_rdd).collect()
        expected_results = [('1', 0.1), ('2', 0.2), ('3', 0.22), ('4', 0.06),
                            ('5', 0.08), ('6', 0.1), ('7', 0.08), ('8', 0.1),
                            ('9', 0.06)]
        self.assertEqual(set(results), set(expected_results))

    def test_likelihood_calculations(self):
        """
        This function tests the calculations of likelihoods given a pair RDD of
        (label, word) where the likelihood is given in form
        ((label,word) likelihood) by counting word counts totally and by label
        """
        dat = [('1', '53'), ('1', '8F'),
               ('1', '48'), ('1', '00'),
               ('1', 'A9'), ('1', '88'),
               ('2', '40'), ('2', '00'),
               ('2', '04'), ('2', '4E'),
               ('2', '00'), ('2', '00'),
               ('3', 'F9'), ('3', '31'),
               ('3', '4F'), ('3', '00'),
               ('3', '1D'), ('3', '99'),
               ('4', '02'), ('4', '47'),
               ('4', 'D5'), ('4', '4F'),
               ('4', '00'), ('4', '00'),
               ('5', '03'), ('5', '05'),
               ('5', 'B5'), ('5', '42'),
               ('5', 'CE'), ('5', '88')]
        test_rdd = self.spark.sparkContext.parallelize(dat, 2)
        results = malware_classifier.calculate_priors(test_rdd).collect()
        expected_results = [(('1', '53'), 1.0), (('1', '8F'), 1.0),
                            (('1', '88'), 0.5), (('3', '31'), 1.0),
                            (('3', '00'), 0.14285714285714285),
                            (('3', '99'), 1.0), (('4', '4F'), 0.5),
                            (('2', '40'), 1.0), (('3', '4F'), 0.5),
                            (('3', '1D'), 1.0), (('4', '02'), 1.0),
                            (('4', '47'), 1.0),
                            (('4', '00'), 0.2857142857142857),
                            (('5', 'CE'), 1.0), (('5', '88'), 0.5),
                            (('1', '48'), 1.0),
                            (('2', '00'), 0.42857142857142855),
                            (('2', '04'), 1.0), (('3', 'F9'), 1.0),
                            (('5', 'B5'), 1.0), (('5', '42'), 1.0),
                            (('1', '00'), 0.14285714285714285),
                            (('1', 'A9'), 1.0), (('2', '4E'), 1.0),
                            (('4', 'D5'), 1.0), (('5', '03'), 1.0),
                            (('5', '05'), 1.0)]
        self.assertEqual(set(results), set(expected_results))

if __name__ == '__main__':
    unittest.main()
