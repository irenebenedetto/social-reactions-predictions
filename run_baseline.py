
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pickle as pkl
import argparse
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_folder', type=str, help='Input/output folder')
    args = parser.parse_args()
    input_folder = args.input_folder

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

    ys_test = pd.read_csv(f'{input_folder}/dataset/ys_test.csv', index_col='id')

    baseline = pd.read_csv(f'{input_folder}/dataset/baseline.csv', index_col='id').loc[ys_test.index]
    ys = pd.read_csv(f'{input_folder}/dataset/ys.csv', index_col='id').loc[ys_test.index]

    results_baseline = {}
    for y_col in y_cols:
        reaction = y_col.split('_')[-1]

        mse = mean_squared_error(ys[y_col], baseline[f"expected_{reaction}"])
        mae = mean_absolute_error(ys[y_col], baseline[f"expected_{reaction}"])

        results_baseline[y_col] = {
            'mse': mse,
            'mae': mae,

        }
        print(reaction)
        print(f'MSE: {round(mse, 6)}\tMAE: {round(mae, 6)}')
        print()

    with open(f"{input_folder}/results_baseline.json", 'w') as f:
        json.dump(results_baseline, f)