import os
import logging
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
import churn_library as cls

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.FileHandler('./logs/churn_library.log', mode='w')
handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

@pytest.fixture
def import_data():
	return cls.import_data

@pytest.fixture
def perform_eda():
	return cls.perform_eda

@pytest.fixture
def encoder_helper():
	return cls.encoder_helper

@pytest.fixture
def perform_feature_engineering():
	return cls.perform_feature_engineering

@pytest.fixture
def train_models():
	return cls.train_models


def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	os.makedirs('./images', exist_ok=True)
	try:
		df = cls.import_data("./data/bank_data.csv")
		perform_eda(df)
		logging.info("Testing perform_eda: SUCCESS")
	except Exception as err:
		logging.error("Testing perform_eda: FAILED")
		raise err

	try:
		assert os.path.isfile('./images/total_trans_ct.png')
		assert os.path.isfile('./images/quant_heatmap.png')
	except AssertionError as err:
		logging.error("Testing perform_eda: images were not saved")
		raise err

def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''
	try: 
		df = cls.import_data("./data/bank_data.csv")
		df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
		cat_columns = df.select_dtypes(include=['str', 'category']).columns.tolist()
		df2 = encoder_helper(df, cat_columns, None)
		logging.info("Testing test_encoder_helper: SUCCESS")
	except Exception as err:
		logging.error("Testing test_encoder_helper: FAILD")
		raise err

	try:
		assert len(df2.columns) != len(df.columns)	
	except AssertionError as err:
		logging.error("Testing test_encoder_helper: fields not created")
		raise err

def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")
		X_train, X_test, y_train, y_test = perform_feature_engineering(df, None)
		logging.info("Testing test_perform_feature_engineering: PASSED")
	except Exception as err:
		logging.error("Testing test_perform_feature_engineering: FAILED")
		raise err

	try:
		assert X_train.shape[0] > 0
		assert X_test.shape[0] > 0
		assert y_train.shape[0] > 0
		assert y_test.shape[0] > 0
		assert X_train.shape[0] == y_train.shape[0]
		assert X_test.shape[0] == y_test.shape[0]
	except AssertionError as err:
		logging.error("Testing test_perform_feature_engineering: outputs are empty")
		raise err


def test_train_models(train_models):
	'''
	test train_models
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")
		X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, None)
		train_models(X_train, X_test, y_train, y_test)
		logging.info("Testing test_train_models: SUCCESS")
	except Exception as err:
		logging.error("Testing test_train_models: FAILED")
		raise err

	try:
		assert os.path.isfile('./models/rfc_model.pkl')
		assert os.path.isfile('./models/logistic_model.pkl')
	except AssertionError as err:
		logging.error("Testing test_train_models: models were not saved")
		raise err


if __name__ == "__main__":
	pass








