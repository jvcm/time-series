import numpy as np

EPSILON = 1E-6



def mean_squared_error(y_true, y_pred):
    y_true = np.asmatrix(y_true + EPSILON).reshape(-1) 
    y_pred = np.asmatrix(y_pred+ EPSILON).reshape(-1)
 
    return np.square(np.subtract(y_true, y_pred)).mean()
 

def mean_absolute_error(y_true, y_pred):
     
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
 
    return np.mean(np.abs(y_true - y_pred))
 

def average_relative_variance(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)
 
    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_pred, mean)).sum()
 
    return error_sup / error_inf
    
def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.abs(_percentage_error(actual, predicted))) * 100
    
def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)
    
    
def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted