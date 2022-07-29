#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pickle as pkl

import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor


os.makedirs('0.results', exist_ok=True)
os.makedirs('1.results', exist_ok=True)
os.makedirs('3.results', exist_ok=True)
os.makedirs('preprocessing', exist_ok=True)

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
    

SAVE = True

ys_test = pd.read_csv('./dataset/ys_test.csv', index_col='id')

baseline = pd.read_csv('./dataset/baseline.csv', index_col='id').loc[ys_test.index]
ys = pd.read_csv('./dataset/ys.csv', index_col='id').loc[ys_test.index]

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


    
with open(f"./results_baseline.json", 'w') as f:
    json.dump(results_baseline, f)


# # TF-IDF (text only)


SAVE = True
X_train = pd.read_csv('./dataset/X_train.csv', index_col='id')
X_test = pd.read_csv('./dataset/X_test.csv', index_col='id')
X_val = pd.read_csv('./dataset/X_val.csv', index_col='id')

ys_train = pd.read_csv('./dataset/ys_train.csv', index_col='id')
ys_test = pd.read_csv('./dataset/ys_test.csv', index_col='id')
ys_val = pd.read_csv('./dataset/ys_val.csv', index_col='id')



# encode text
vectorizer = TfidfVectorizer()

features = [
            'type',
]

sp_train = vectorizer.fit_transform(X_train['content'])
sp_val = vectorizer.transform(X_val['content'])
sp_test = vectorizer.transform(X_test['content'])


if SAVE:
    with open('./preprocessing/feature_list.json', 'w') as f:
        json.dump(vectorizer.get_feature_names() + features, f)
    with open('./preprocessing/tfidf_vectorizer.pkl', 'wb') as f:
        pkl.dump(vectorizer, f)


models = [
    LinearRegression, # lineare sempre
    MLPRegressor,
    AdaBoostRegressor
]

model_names = [
    'linear_regression',
    'mlp_regression',
    'adaboost_regression',
]

parameters = [
    {
        'fit_intercept': [True, False]
     },
    {
        'hidden_layer_sizes': [10, 20],
        'learning_rate': ['adaptive'],
        'max_iter': [1, 2],
        'verbose': [True],
        'tol': [1e-3]
    },
    {
        'n_estimators': [5, 10]
    },
    
]


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

best_predictor = None
best_params = None
best_scores_val = {'mae': 1e20 }



# train and validate
predictions = {}
for model_name, model, param in zip(model_names, models, parameters):
    os.makedirs(f'./0.results/{model_name}', exist_ok=True)
    print(f'Predicting {y_col}')

    for y_col in y_cols:
    
        os.makedirs(f'./0.results/{model_name}/{y_col}', exist_ok=True)

        print(f'Model {model_name}')

        for param_grid in ParameterGrid(param):
            estimator = model(**param_grid)
            estimator.fit(sp_train, ys_train[y_col])

            y_pred = estimator.predict(sp_val)
            mse = mean_squared_error(ys_val[y_col], y_pred)
            mae = mean_absolute_error(ys_val[y_col], y_pred)

            if mae < best_scores_val['mae']:
                best_predictor = estimator
                best_params = param_grid
                best_scores_val = {
                    'mse': mse,
                    'mae': mae
                }


        print(f'\nBest score on validation set')
        print(f'MSE: {round(best_scores_val["mse"], 3)}\tMAE: {round(best_scores_val["mae"], 3)}')
        if SAVE:
            with open(f"./0.results/{model_name}/{y_col}/best_estimator.pkl", 'wb') as f:
                pkl.dump(best_predictor, f)

            with open(f"./0.results/{model_name}/{y_col}/best_scores_val.json", 'w') as f:
                json.dump(best_scores_val, f)

            with open(f"./0.results/{model_name}/{y_col}/best_params.pkl", 'wb') as f:
                pkl.dump(best_params, f)

        y_pred = estimator.predict(sp_test)
        predictions[f"pred_{y_col.split('_')[-1]}"] = y_pred

        mse = mean_squared_error(ys_test[y_col], y_pred)
        mae = mean_absolute_error(ys_test[y_col], y_pred)

        test_scores = {
            'mse': mse,
            'mae': mae,
        }

        print(f'Score on test set')
        print(f'MSE: {round(test_scores["mse"], 3)}\tMAE: {round(test_scores["mae"], 3)}')
        print()
        with open(f"./0.results/{model_name}/{y_col}/test_scores.json", 'w') as f:
            json.dump(test_scores, f)

    predictions = pd.DataFrame(predictions)
    predictions['id'] = X_test.index
    predictions.set_index('id', inplace=True)

    output = pd.concat([X_test, predictions], axis = 1)
    output.reset_index(inplace=True)
    ys_test.reset_index(inplace=True)
    output = pd.concat([output,  ys_test], axis = 1)

    output.to_csv(f'./0.results/{model_name}/test_predictions.csv', index=None)
