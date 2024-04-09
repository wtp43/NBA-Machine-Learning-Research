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
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing




def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_test_split(data):
    # cutoff = math.floor(len(data) * 0.7)
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
        data = pd.read_csv('nba_processed_features + elo.csv')
    except Exception as e:
        logger.exception(
            "Unable to load CSV. Error: %s", e
        )

    columns = data.columns
    p = re.compile('prev.*ema')
    # features = ['season', 'home_ml','home_win', 'away_ml', 'prev_home_elo', 'prev_away_elo'] 
    # features += [c for c in columns if p.match(c)]
    w = [20,14,8]

    features = []

    player_features = ['ts_pct',
        'efg_pct', 'threepar', 'ftr', 'orb_pct', 'drb_pct', 'trb_pct',
        'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct', 'ortg', 'drtg',
        'fg', 'fga', 'fg_pct', 'threep', 'threepa',
        'threep_pct', 'ft', 'fta', 'ft_pct', 'orb', 'drb', 'trb', 'ast', 'stl',
        'blk', 'tov', 'pf', 'pts', 'pts_avg', 'orb_avg', 'drb_avg', 'trb_avg',
        'ast_avg', 'stl_avg', 'blk_avg', 'tov_avg', 'pf_avg']

    w = 10
    fav_player_ewm_features = []
    und_player_ewm_features = []
    for i in range(8):
        fav_player_ewm_features += [f'fav_p{i}_{f}_ewm_{w}' for f in player_features]
        und_player_ewm_features += [f'und_p{i}_{f}_ewm_{w}' for f in player_features]

    w = 14
    fav_player_ewm_fatigue = []
    und_player_ewm_fatigue = []
    for i in range(8):
        fav_player_ewm_fatigue += [f'fav_p{i}_{f}_ewm_{w}' for f in player_features]
        und_player_ewm_fatigue += [f'und_p{i}_{f}_ewm_{w}' for f in player_features]

    streak = ['prev_favorite_win_streak', 
                    'prev_favorite_home_streak', 
                    'prev_underdog_win_streak', 
                    'prev_underdog_home_streak']

    elo = ['prev_favorite_elo', 'prev_underdog_elo']
    # 5, 8, 12
    window_size = 8
    ema_favorite_features = \
            [f'prev_favorite_pts_ema{window_size}',     
            f'prev_favorite_bpm_ema{window_size}',      
            f'prev_favorite_fg_ema{window_size}',        
            f'prev_favorite_fg_pct_ema{window_size}',  
            f'prev_favorite_3p_ema{window_size}',       
            f'prev_favorite_3p_pct_ema{window_size}',   
            f'prev_favorite_ft_ema{window_size}',       
            f'prev_favorite_ft_pct_ema{window_size}',    
            f'prev_favorite_orb_ema{window_size}',       
            f'prev_favorite_orb_pct_ema{window_size}',  
            f'prev_favorite_drb_ema{window_size}',      
            f'prev_favorite_drb_pct_ema{window_size}',   
            f'prev_favorite_trb_ema{window_size}',       
            f'prev_favorite_trb_pct_ema{window_size}',   
            f'prev_favorite_tov_ema{window_size}',       
            f'prev_favorite_tov_pct_ema{window_size}',  
            f'prev_favorite_ast_ema{window_size}',       
            f'prev_favorite_ast_pct_ema{window_size}',   
            f'prev_favorite_stl_ema{window_size}',       
            f'prev_favorite_stl_pct_ema{window_size}',   
            f'prev_favorite_blk_ema{window_size}',       
            f'prev_favorite_blk_pct_ema{window_size}',   
            f'prev_favorite_drtg_ema{window_size}',      
            f'prev_favorite_ortg_ema{window_size}',      
            f'prev_favorite_efg_pct_ema{window_size}',  
            f'prev_favorite_pace_ema{window_size}']

    ema_underdog_features = [f.replace('favorite','underdog') for f in ema_favorite_features]

    features += fav_player_ewm_features + und_player_ewm_features + \
                streak + elo + ema_favorite_features + ema_underdog_features
    data.dropna(inplace=True)
    from imblearn.over_sampling import RandomOverSampler
    mm_scaler = preprocessing.MinMaxScaler()

    train, test = train_test_split(data)

    X_train = train[features]
    X_test = test[features]

    X_train = mm_scaler.fit_transform(X_train)
    X_test = mm_scaler.fit_transform(X_test)

    y_train = train['favorite_won']
    y_test = test['favorite_won']

    # oversample training data
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_train, y_train = oversample.fit_resample(X_train, y_train)


    # Split the data into training and test sets. (0.75, 0.25) split.

    train_x = X_train
    test_x = X_test
    train_y = y_train
    test_y = y_test

    with mlflow.start_run():
        xgb_model = xgb.XGBClassifier(tree_method = 'gpu_hist', 
                              gpu_id = 0, 
                              eval_metric='logloss', 
                              random_state = 1,
                              use_label_encoder=False)
        tscv = TimeSeriesSplit(n_splits=3)
        # params = {
        # 'min_child_weight':[3,5,10],
        # 'alpha':[10,50],
        # 'gamma':[0.1,0.2,0.3,1],
        # 'lambda':[1,10],
        # 'subsample':[0.6,0.8, 1.0],
        # 'colsample_bytree':[0.6,0.8, 1.0],
        # 'max_depth':[6,10,15,20],
        # 'n_estimators':[10, 100,200],
        # 'learning_rate':[0.01,0.1,0.2]
        # }

        params = {
        'min_child_weight':[3,5],
        'alpha':[10],
        'gamma':[0.1],
        'lambda':[1],
        'subsample':[1.0],
        'colsample_bytree':[1.0],
        'max_depth':[6,20],
        'n_estimators':[10,100],
        'learning_rate':[0.001, 0.01]
        }

        # params = {
        # 'min_child_weight':[3],
        # 'alpha':[10],
        # 'gamma':[0.1],
        # 'lambda':[1],
        # 'subsample':[0.6],
        # 'colsample_bytree':[0.6],
        # 'max_depth':[6],
        # 'n_estimators':[10],
        # 'learning_rate':[0.1]
        # }

        grid_search = GridSearchCV(estimator = xgb_model,
                                    cv = tscv, 
                                    scoring = 'accuracy',
                                    param_grid = params,
                                    n_jobs = 15,
                                    verbose=1)
        grid_search.fit(train_x, train_y)
        print("Xgboost model")
        print(" Results from Grid Search " )
        print("\n The best estimator across ALL searched params:\n",grid_search.best_estimator_)
        print("\n The best score across ALL searched params:\n",grid_search.best_score_)
        print("\n The best parameters across ALL searched params:\n",grid_search.best_params_)
    


        predictions = grid_search.best_estimator_.predict(test_x)
        predictions = pd.Series(predictions, name='xgb_favorite_won')
        pred_df = pd.concat([test_y.reset_index(drop=True), 
                        predictions.reset_index(drop=True),
                    # test['home_ml'].reset_index(drop=True),
                    # test['away_ml'].reset_index(drop=True),
                    ], axis=1)
        
        pred_df['correct_pred'] = pred_df.apply(lambda x: 
            1 if x['favorite_won'] == x['xgb_favorite_won'] else 0, axis = 1)
        
        acc = pred_df['correct_pred'].sum()/len(pred_df.index)
        pred_df.to_csv('xgb_predictions.csv', header=True, index=False)
        print(f'Accuracy: {acc}')
        mlflow.log_metric("Accuracy", acc)
        confusion_mat = confusion_matrix(pred_df['favorite_won'], pred_df['xgb_favorite_won'])
        print(confusion_mat)
        f1 = f1_score(pred_df['favorite_won'], pred_df['xgb_favorite_won'])
        print(f'F1 Score: {f1}')
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        # if tracking_url_type_store != "file":

        #     # Register the model
        #     # There are other ways to use the Model Registry, which depends on the use case,
        #     # please refer to the doc for more information:
        #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        #     mlflow.sklearn.log_model(xgb_model, "model", registered_model_name="z")
        # else:
        #     mlflow.sklearn.log_model(xgb_model, "model")
        f = open("xgb_best_params.txt", "w")
        f.write(str(grid_search.best_params_))
        f.write(f'\n Accuracy\n: {acc}')
        f.close()