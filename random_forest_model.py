import pandas as pd
import numpy as np

import os
import warnings
import sys
import math
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.metrics import confusion_matrix, f1_score
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_test_split(data):
    #cutoff = math.floor(len(data) * 0.7)
    # train = data[data.index < cutoff].copy()
    # test = data[data.index >= cutoff].copy()
    train = data[data.season < 2020].copy()
    test = data[data.season >= 2020].copy()
    return train, test 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    try:
        match_data = pd.read_csv('modern_matches.csv')
    except Exception as e:
        logger.exception(
            "Unable to load CSV. Error: %s", e
        )
    
    columns = match_data.columns
    data = match_data.copy()
    match_data['home_win'] = data['movl'].map(lambda x: 0 if x < 0 else 1)
    p = re.compile('prev.*ema')
    features = ['season', 'home_ml', 'movl', 
                #'prev_home_elo', 'prev_away_elo', 
                'spread', 'home_spread'
                ] 
    features += [c for c in columns if p.match(c)]
    data = data[features]
    data.dropna(inplace=True)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train = train.drop(["season"], axis=1)
    test = test.drop(["season"], axis=1)

    train_x = train.drop(["movl"], axis=1)
    test_x = test.drop(["movl"], axis=1)
    train_y = train[["movl"]]
    test_y = test[["movl"]]

    tscv = TimeSeriesSplit(n_splits=3)
    
    ## Define Grid 
    params = { 
        'n_estimators': [50, 100, 500],
        #'min_samples_leaf': [50, 100,150], 
        'max_features': ['sqrt', 'log2'],
        'max_depth' : [3, 5, 7, 9],
        'random_state' : [18]
    }
    ## Grid Search function
    grid_search = GridSearchCV(estimator = RandomForestRegressor(),
                                cv = tscv, 
                                param_grid = params,
                                n_jobs = -2)
    grid_search.fit(train_x, train_y.values.ravel())
    print("Random forest model")
    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",grid_search.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_search.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid_search.best_params_)

    test['home_win'] = test['movl'].map(lambda x: 0 if x < 0 else 1)
    predictions = grid_search.best_estimator_.predict(test_x)
    predictions = pd.Series(predictions, name='rfr_movl')
    pred_df = pd.concat([test_y.reset_index(drop=True), 
                    predictions.reset_index(drop=True),
                #test['home_ml'].reset_index(drop=True),
                test['home_win'].reset_index(drop=True)
                ], axis=1)
    
    pred_df['rfr_prediction'] = pred_df.apply(lambda x: 
        1 if x['rfr_movl'] > 0 else 0, axis = 1)
    pred_df['correct_pred'] = pred_df.apply(lambda x: 
        1 if x['home_win'] == x['rfr_prediction'] else 0, axis = 1)
    
    acc = pred_df['correct_pred'].sum()/len(pred_df.index)
    pred_df.to_csv('rfr_predictions.csv', header=True, index=False)
    print(f'Accuracy: {acc}')
    mlflow.log_metric("Accuracy", acc)
    confusion_mat = confusion_matrix(pred_df['home_win'], pred_df['rfr_prediction'])
    print(confusion_mat)
    f1 = f1_score(pred_df['home_win'], pred_df['rfr_prediction'])
    print(f'F1 Score: {f1}')
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
