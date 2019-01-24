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

    def test_toxenization_and_preproc(self):
        dat = ['00401060 53 8F 48 00 A9 88 40 00 04 4E 00 00 F9 31 4F 00',
               '00401070 1D 99 02 47 D5 4F 00 00 03 05 B5 42 CE 88 65 43']
        test_rdd = self.spark.sparkContext.parallelize(dat, 2)
        results = malware_classifier.tokenize_and_preproc(test_rdd).collect()
        expected_results = ['53', '8F', '48', '00', 'A9', '88', '40', '00',
                            '04', '4E', '00', '00', 'F9', '31', '4F', '00',
                            '1D', '99', '02', '47', 'D5', '4F', '00', '00',
                            '03', '05', 'B5', '42', 'CE', '88', '65', '43']
        self.assertEqual(set(results), set(expected_results))


if __name__ == '__main__':
    unittest.main()
