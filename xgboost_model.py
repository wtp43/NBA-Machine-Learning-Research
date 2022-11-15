import pandas as pd
import numpy as np

import os
import warnings
import sys
import math
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.metrics import confusion_matrix, f1_score
import xgboost as xgb
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
        data = pd.read_csv('modern_matches.csv')
    except Exception as e:
        logger.exception(
            "Unable to load CSV. Error: %s", e
        )
    
    columns = data.columns
    data['home_win'] = data['movl'].map(lambda x: 0 if x < 0 else 1)
    p = re.compile('prev.*ema')
    features = ['season', 'home_ml','home_win', 'away_ml', 'prev_home_elo', 'prev_away_elo'] 
    features += [c for c in columns if p.match(c)]
    data = data[features]
    data.dropna(inplace=True)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train = train.drop(["season", 'away_ml'], axis=1)
    test = test.drop(["season"], axis=1)
    train_x = train.drop(["home_win"], axis=1)
    test_x = test.drop(["home_win", 'away_ml'], axis=1)
    train_y = train[["home_win"]]
    test_y = test[["home_win"]]

    with mlflow.start_run():
        xgb_model = xgb.XGBClassifier(tree_method = 'gpu_hist', 
                              gpu_id = 0, 
                              eval_metric='logloss', 
                              random_state = 1,
                              use_label_encoder=False)
        tscv = TimeSeriesSplit(n_splits=3)
        # params = {
        # 'min_child_weight':[3,5],
        # 'alpha':[10,50],
        # 'gamma':[0.1,0.2,0.3,0.4,1],
        # 'lambda':[1,10],
        # 'subsample':[0.6, 0.8, 1.0],
        # 'colsample_bytree':[0.6, 0.8, 1.0],
        # 'max_depth':[6,10,20],
        # 'n_estimators':[10,50,100],
        # 'learning_rate':[0.01,0.1,0.2, 0.3]
        # }

        # params = {
        # 'min_child_weight':[3,5],
        # 'alpha':[10,50],
        # 'gamma':[0.1,0.2,1],
        # 'lambda':[1,10],
        # 'subsample':[0.6,1.0],
        # 'colsample_bytree':[0.6,1.0],
        # 'max_depth':[6,20],
        # 'n_estimators':[10,100],
        # 'learning_rate':[0.1, 0.3]
        # }

        params = {
        'min_child_weight':[3],
        'alpha':[10],
        'gamma':[0.1],
        'lambda':[1],
        'subsample':[0.6],
        'colsample_bytree':[0.6],
        'max_depth':[6],
        'n_estimators':[10],
        'learning_rate':[0.1]
        }

        grid_search = GridSearchCV(estimator = xgb_model,
                                    cv = tscv, 
                                    scoring = 'accuracy',
                                    param_grid = params,
                                    n_jobs = -1,
                                    verbose=1)
        grid_search.fit(train_x, train_y)
        print("Xgboost model")
        print(" Results from Grid Search " )
        print("\n The best estimator across ALL searched params:\n",grid_search.best_estimator_)
        print("\n The best score across ALL searched params:\n",grid_search.best_score_)
        print("\n The best parameters across ALL searched params:\n",grid_search.best_params_)

        predictions = grid_search.best_estimator_.predict(test_x)
        predictions = pd.Series(predictions, name='xgb_home_win')
        pred_df = pd.concat([test_y.reset_index(drop=True), 
                        predictions.reset_index(drop=True),
                    test['home_ml'].reset_index(drop=True),
                    test['away_ml'].reset_index(drop=True),
                    ], axis=1)
        
        pred_df['correct_pred'] = pred_df.apply(lambda x: 
            1 if x['home_win'] == x['xgb_home_win'] else 0, axis = 1)
        
        acc = pred_df['correct_pred'].sum()/len(pred_df.index)
        pred_df.to_csv('xgb_predictions.csv', header=True, index=False)
        print(f'Accuracy: {acc}')
        mlflow.log_metric("Accuracy", acc)
        confusion_mat = confusion_matrix(pred_df['home_win'], pred_df['xgb_home_win'])
        print(confusion_mat)
        f1 = f1_score(pred_df['home_win'], pred_df['xgb_home_win'])
        print(f'F1 Score: {f1}')
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(xgb_model, "model", registered_model_name="ElasticnetModel")
        else:
            mlflow.sklearn.log_model(xgb_model, "model")
