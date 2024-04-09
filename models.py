import pandas as pd
import numpy as np

import os
import warnings
import sys
import math
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

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
    train = data[data.season < 2021].copy()
    test = data[data.season >= 2021].copy()
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
    p = re.compile('prev.*ema')
    features = ['season', 'home_ml', 'away_ml', 'movl', 'prev_home_elo', 'prev_away_elo'] 
    features += [c for c in columns if p.match(c)]
    data = data[features]
    data.dropna(inplace=True)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train = train.drop(["season", 'away_ml'], axis=1)
    test = test.drop(["season"], axis=1)
    train_x = train.drop(["movl"], axis=1)
    test_x = test.drop(["movl", 'away_ml'], axis=1)
    train_y = train[["movl"]]
    test_y = test[["movl"]]

    print(train_x.columns)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predictions = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predictions)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = pd.Series(predictions, name='enet_movl')
        pred_df = pd.concat([test_y.reset_index(drop=True), 
                        predictions.reset_index(drop=True),
                    test['home_ml'].reset_index(drop=True),
                    test['away_ml'].reset_index(drop=True)], axis=1)
        pred_df['home_win'] = pred_df['movl'].map(lambda x: 0 if x < 0 else 1)
        pred_df['enet_home_win'] = pred_df['enet_movl'].map(lambda x: 0 if x < 0 else 1)
        pred_df['correct_pred'] = pred_df.apply(lambda x: 
            1 if x['home_win'] == x['enet_home_win'] else 0, axis = 1)
        acc = pred_df['correct_pred'].sum()/len(pred_df.index)
        pred_df.to_csv('enet_predictions.csv', header=True, index=False)
        print(f'Accuracy: {acc}')
        mlflow.log_metric("Accuracy", acc)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetModel")
        else:
            mlflow.sklearn.log_model(lr, "model")
