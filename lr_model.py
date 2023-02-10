import pandas as pd
import numpy as np
import pickle

import warnings
import sys
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression 
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
        data = pd.read_csv('modern_matches.csv')
    except Exception as e:
        logger.exception(
            "Unable to load CSV. Error: %s", e
        )
    
    columns = data.columns
    p = re.compile('prev.*ema')
    data['home_win'] = data['movl'].map(lambda x: 0 if x < 0 else 1)
    features = ['season', 'home_ml', 'home_win', 
                #'prev_home_elo', 'prev_away_elo',
                'spread', 'home_spread',
                ] 
#features += [c for c in columns if p.match(c)]
    data = data[features]
    data.dropna(inplace=True)
    

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train = train.drop(["season"], axis=1)
    test = test.drop(["season"], axis=1)
    train_x = train.drop(["home_win"], axis=1)
    test_x = test.drop(["home_win"], axis=1)
    train_y = train[["home_win"]]
    test_y = test[["home_win"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = LogisticRegression(random_state=42)
        lr.fit(train_x, train_y)

        predictions = lr.predict_proba(test_x)
        predictions = [i[1] for i in predictions]

        predictions = pd.Series(predictions, name='lr_home_proba')
        pred_df = pd.concat([test_y.reset_index(drop=True), 
                        predictions.reset_index(drop=True),
                    test['home_ml'].reset_index(drop=True)], axis=1)
        pred_df['lr_home_win'] = pred_df['lr_home_proba'].map(lambda x: 0 if x < 0.53 else 1)
        pred_df['correct_pred'] = pred_df.apply(lambda x: 
            1 if x['home_win'] == x['lr_home_win'] else 0, axis = 1)
        acc = pred_df['correct_pred'].sum()/len(pred_df.index)
        pred_df.to_csv('lr_predictions.csv', header=True, index=False)
        print(f'Accuracy: {acc}')
        mlflow.log_metric("Accuracy", acc)

        confusion_mat = confusion_matrix(pred_df['home_win'], pred_df['lr_home_win'])
        print(confusion_mat)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        f1 = f1_score(pred_df['home_win'], pred_df['lr_home_win'])
        print(f'F1 Score: {f1}')

        print(pred_df.head())

        importance =lr.coef_[0]
        
        filename = 'final_lr_model.sav'
        pickle.dump(lr, open(filename, 'wb'))

        # summarize feature importance
        # for i,v in enumerate(importance):
        #     print('Feature: %0d, Score: %.5f' % (i,v))
