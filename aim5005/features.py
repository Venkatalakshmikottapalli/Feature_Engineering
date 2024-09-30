import numpy as np
from typing import List, Tuple
### YOU MAY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum

        # Corrected Code
        return (x-self.minimum)/diff_max_min
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Ensure the input x is a numpy array. Convert if not, and raise an error if it can't be cast.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected input to be a numpy array or list"
        return x

    def fit(self, x: np.ndarray) -> None:
        """
        Compute the mean and standard deviation for each feature.
        """
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize the data by subtracting the mean and dividing by the standard deviation.
        """
        x = self._check_is_array(x)
        
        # Handle case where std is 0 to avoid division by zero
        self.std[self.std == 0] = 1
        
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        """
        self.fit(x)
        return self.transform(x)
class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        
    def fit(self, x: List) -> None:
        x = np.array(x)
        self.classes_ = np.sort(np.unique(x))
        
    def transform(self, x: List) -> np.ndarray:
        x = np.array(x)
        return np.array([np.where(self.classes_ == label)[0][0] for label in x])
    
    def fit_transform(self, x: List) -> np.ndarray:
        self.fit(x)
        return self.transform(x)