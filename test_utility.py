import pytest
import pandas as pd
import numpy as np
from prediction_demo import data_preparation,data_split,train_model,eval_model

@pytest.fixture
def housing_data_sample():
    return pd.DataFrame(
      data ={
      'price':[13300000,12250000],
      'area':[7420,8960],
    	'bedrooms':[4,4],	
      'bathrooms':[2,4],	
      'stories':[3,4],	
      'mainroad':["yes","yes"],	
      'guestroom':["no","no"],	
      'basement':["no","no"],	
      'hotwaterheating':["no","no"],	
      'airconditioning':["yes","yes"],	
      'parking':[2,3],
      'prefarea':["yes","no"],	
      'furnishingstatus':["furnished","unfurnished"]}
    )

def test_data_preparation(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    # Target and datapoints has same length
    assert feature_df.shape[0]==len(target_series)

    #Feature only has numerical values
    assert feature_df.shape[1] == feature_df.select_dtypes(include=(np.number,np.bool_)).shape[1]

@pytest.fixture
def feature_target_sample(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    return (feature_df, target_series)

def test_data_split(feature_target_sample):
    return_tuple = data_split(*feature_target_sample)
    
    # Test if the length of return_tuple is 4
    assert len(return_tuple) == 4, "data_split should return a tuple of 4 elements"
    
    # Additional tests to ensure the correct structure of the returned data
    X_train, X_test, y_train, y_test = return_tuple
    
    # Test if the returned elements are of the correct type
    assert isinstance(X_train, pd.DataFrame), "X_train should be a pandas DataFrame"
    assert isinstance(X_test, pd.DataFrame), "X_test should be a pandas DataFrame"
    assert isinstance(y_train, pd.Series), "y_train should be a pandas Series"
    assert isinstance(y_test, pd.Series), "y_test should be a pandas Series"
    
    # Test if the shapes of the returned elements are consistent
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train should have the same number of samples"
    assert X_test.shape[0] == y_test.shape[0], "X_test and y_test should have the same number of samples"
    
    # Test if the total number of samples is preserved
    total_samples = len(feature_target_sample[0])
    assert X_train.shape[0] + X_test.shape[0] == total_samples, "The sum of train and test samples should equal the total number of samples"