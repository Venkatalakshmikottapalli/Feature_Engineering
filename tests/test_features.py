from aim5005.features import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import unittest
from unittest.case import TestCase

### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    # Test cases for Minmax Scaler
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)

    # Test cases for standard scaler  
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    # Custom test for standard scaler: This test case will test the StandardScaler when applied to a dataset with zero variance.When the variance is zero, the StandardScaler should transform those values to 0.
    def test_standard_scaler_zero_variance(self):
        data = [[1, 1], [1, 1], [1, 1], [1, 1]]
        expected = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not handle zero variance correctly. Expect {}. Got: {}".format(expected, result)
    
    # These are test cases for LabelEncoder
    def test_initialize_label_encoder(self):
        encoder = LabelEncoder()
        assert isinstance(encoder, LabelEncoder), "encoder is not a LabelEncoder object"
        
    def test_label_encoder_fit(self):
        labels = ["rose", "tulip", "daisy"]
        encoder = LabelEncoder()
        encoder.fit(labels)
        expected_classes = np.array(["daisy", "rose", "tulip"])
        assert (encoder.classes_ == expected_classes).all(), "encoder fit does not set expected classes {}. Got {}".format(expected_classes, encoder.classes_)
        
    def test_label_encoder_transform(self):
        labels = ["rose", "tulip", "daisy", "rose", "tulip"]
        encoder = LabelEncoder()
        encoder.fit(labels)
        expected_encoded = np.array([1, 2, 0, 1, 2])
        result = encoder.transform(labels)
        assert (result == expected_encoded).all(), "encoder transform does not return expected encoded values {}. Got {}".format(expected_encoded, result)
        
    def test_label_encoder_fit_transform(self):
        labels = ["rose", "tulip", "daisy", "rose", "tulip"]
        encoder = LabelEncoder()
        expected_encoded = np.array([1, 2, 0, 1, 2])
        result = encoder.fit_transform(labels)
        assert (result == expected_encoded).all(), "encoder fit_transform does not return expected encoded values {}. Got {}".format(expected_encoded, result)
    
if __name__ == '__main__':
    unittest.main()