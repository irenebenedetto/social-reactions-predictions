import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pickle as pkl
import argparse
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm 


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_folder', type=str, help='Output folder')
    parser.add_argument('--input_data_folder', type=str, help='Input folder')
    args = parser.parse_args()


	TEST_SIZE = 0.2
	df = pd.read_csv(f'{args.input_data_folder}/data.csv', index_col='id')


	df.drop('Unnamed: 0', inplace=True, axis = 1)

	df['date'] = df['date'].astype('datetime64[ns]')
	df['type'] = df['type'].astype('category')
	df['name'] = df['name'].astype('category')

	df[['type']] = df[['type']].apply(lambda x: x.cat.codes)
	df[['name']] = df[['name']].apply(lambda x: x.cat.codes)

	df['day'] = df.date.dt.day
	df['month'] = df.date.dt.month
	df['year'] = df.date.dt.year
	df['hour'] = df.date.dt.hour
	df['minute'] = df.date.dt.minute

	df.drop('date', inplace=True, axis = 1)

	y_cols = [
	          'actual_angryCount', 
	          'actual_careCount', 
	          'actual_hahaCount',
	          'actual_likeCount', 
	          'actual_loveCount', 
	          'actual_sadCount',
	          'actual_thankfulCount', 
	          'actual_wowCount', 
	          'actual_commentCount',
	          'actual_shareCount'
	]
	x_cols = [
	          'name',
	          'type',
	          'content',
	          'subscriberCount',
	          'day',
	          'year',
	          'month',
	          'hour',
	          'minute'
	]
	baselines_cols = [
	          'expected_angryCount', 
	          'expected_careCount',
	          'expected_hahaCount', 
	          'expected_likeCount', 
	          'expected_loveCount',
	          'expected_sadCount', 
	          'expected_thankfulCount', 
	          'expected_wowCount',
	          'expected_commentCount', 
	          'expected_shareCount',
	]
	X = df[x_cols]
	ys = df[y_cols]
	baseline = df[x_cols + baselines_cols + y_cols]


	X.to_csv(f'{args.output_folder}/X.csv')
	ys.to_csv(f'{args.output_folder}/ys.csv')
	baseline.to_csv(f'{args.output_folder}/baseline.csv')


	TEST_SIZE = 0.2
	DEV_SIZE = 0.1

	X_train, X_test, ys_train, ys_test = train_test_split(X, ys, test_size=(TEST_SIZE + DEV_SIZE))
	X_val, X_test, ys_val, ys_test = train_test_split(X_test, ys_test, test_size=TEST_SIZE/(TEST_SIZE + DEV_SIZE))



	X_train.to_csv(f'{args.output_folder}/X_train.csv')
	X_test.to_csv(f'{args.output_folder}/X_test.csv')
	X_val.to_csv(f'{args.output_folder}/X_val.csv')

	ys_train.to_csv(f'{args.output_folder}/ys_train.csv')
	ys_test.to_csv(f'{args.output_folder}/ys_test.csv')
	ys_val.to_csv(f'{args.output_folder}/ys_val.csv')

	print(f"X_train shape: {X_train.shape} ({ round(X_train.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0]) ,2) }%)")
	print(f"X_val shape: {X_val.shape} ({ round(X_val.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0]) ,2) }%)")
	print(f"X_test shape: {X_test.shape} ({ round(X_test.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0]) ,2) }%)")
