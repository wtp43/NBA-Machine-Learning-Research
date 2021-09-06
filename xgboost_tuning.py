

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as pl
import xgboost as xgb
import sklearn.metrics
from matplotlib import pyplot
import math
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as pl
import xgboost as xgb
import pickle


def main():

	window_size = 4
	hth_window_size = 3

	ema_h_features = [(f'prev_home_pts_ema{window_size}',       f'post_home_pts_ema{window_size}'),
					(f'prev_home_bpm_ema{window_size}',       f'post_home_bpm_ema{window_size}'),
					(f'prev_home_fg_ema{window_size}',        f'post_home_fg_ema{window_size}'),
					(f'prev_home_fg_pct_ema{window_size}',    f'post_home_fg_pct_ema{window_size}'),
					(f'prev_home_3p_ema{window_size}',        f'post_home_3p_ema{window_size}'),
					(f'prev_home_3p_pct_ema{window_size}',    f'post_home_3p_pct_ema{window_size}'),
					(f'prev_home_ft_ema{window_size}',        f'post_home_ft_ema{window_size}'),
					(f'prev_home_ft_pct_ema{window_size}',    f'post_home_ft_pct_ema{window_size}'),
					(f'prev_home_orb_ema{window_size}',       f'post_home_orb_ema{window_size}'),
					(f'prev_home_orb_pct_ema{window_size}',   f'post_home_orb_pct_ema{window_size}'),
					(f'prev_home_drb_ema{window_size}',       f'post_home_drb_ema{window_size}'),
					(f'prev_home_drb_pct_ema{window_size}',   f'post_home_drb_pct_ema{window_size}'),
					(f'prev_home_trb_ema{window_size}',       f'post_home_trb_ema{window_size}'),
					(f'prev_home_trb_pct_ema{window_size}',   f'post_home_trb_pct_ema{window_size}'),
					(f'prev_home_tov_ema{window_size}',       f'post_home_tov_ema{window_size}'),
					(f'prev_home_tov_pct_ema{window_size}',   f'post_home_tov_pct_ema{window_size}'),
					(f'prev_home_ast_ema{window_size}',       f'post_home_ast_ema{window_size}'),
					(f'prev_home_ast_pct_ema{window_size}',   f'post_home_ast_pct_ema{window_size}'),
					(f'prev_home_stl_ema{window_size}',       f'post_home_stl_ema{window_size}'),
					(f'prev_home_stl_pct_ema{window_size}',   f'post_home_stl_pct_ema{window_size}'),
					(f'prev_home_blk_ema{window_size}',       f'post_home_blk_ema{window_size}'),
					(f'prev_home_blk_pct_ema{window_size}',   f'post_home_blk_pct_ema{window_size}'),
					(f'prev_home_drtg_ema{window_size}',      f'post_home_drtg_ema{window_size}'),
					(f'prev_home_ortg_ema{window_size}',      f'post_home_ortg_ema{window_size}'),
					(f'prev_home_efg_pct_ema{window_size}',   f'post_home_efg_pct_ema{window_size}')]

	ema_a_features = [(f[0].replace('home','away'), f[1].replace('home','away')) for f in ema_h_features]
	sma_h_features = [f[0].replace('ema','sma') for f in ema_h_features]
	sma_a_features = [f.replace('home','away') for f in sma_h_features]

	ema_h_hth_features = [(f'prev_hth_home_pts_ema{hth_window_size}',       f'post_hth_home_pts_ema{hth_window_size}'),
						(f'prev_hth_home_bpm_ema{hth_window_size}',       f'post_hth_home_bpm_ema{hth_window_size}'),
						(f'prev_hth_home_fg_ema{hth_window_size}',        f'post_hth_home_fg_ema{hth_window_size}'),
						(f'prev_hth_home_fg_pct_ema{hth_window_size}',    f'post_hth_home_fg_pct_ema{hth_window_size}'),
						(f'prev_hth_home_3p_ema{hth_window_size}',        f'post_hth_home_3p_ema{hth_window_size}'),
						(f'prev_hth_home_3p_pct_ema{hth_window_size}',    f'post_hth_home_3p_pct_ema{hth_window_size}'),
						(f'prev_hth_home_ft_ema{hth_window_size}',        f'post_hth_home_ft_ema{hth_window_size}'),
						(f'prev_hth_home_ft_pct_ema{hth_window_size}',    f'post_hth_home_ft_pct_ema{hth_window_size}'),
						(f'prev_hth_home_orb_ema{hth_window_size}',       f'post_hth_home_orb_ema{hth_window_size}'),
						(f'prev_hth_home_orb_pct_ema{hth_window_size}',   f'post_hth_home_orb_pct_ema{hth_window_size}'),
						(f'prev_hth_home_drb_ema{hth_window_size}',       f'post_hth_home_drb_ema{hth_window_size}'),
						(f'prev_hth_home_drb_pct_ema{hth_window_size}',   f'post_hth_home_drb_pct_ema{hth_window_size}'),
						(f'prev_hth_home_trb_ema{hth_window_size}',       f'post_hth_home_trb_ema{hth_window_size}'),
						(f'prev_hth_home_trb_pct_ema{hth_window_size}',   f'post_hth_home_trb_pct_ema{hth_window_size}'),
						(f'prev_hth_home_tov_ema{hth_window_size}',       f'post_hth_home_tov_ema{hth_window_size}'),
						(f'prev_hth_home_tov_pct_ema{hth_window_size}',   f'post_hth_home_tov_pct_ema{hth_window_size}'),
						(f'prev_hth_home_ast_ema{hth_window_size}',       f'post_hth_home_ast_ema{hth_window_size}'),
						(f'prev_hth_home_ast_pct_ema{hth_window_size}',   f'post_hth_home_ast_pct_ema{hth_window_size}'),
						(f'prev_hth_home_stl_ema{hth_window_size}',       f'post_hth_home_stl_ema{hth_window_size}'),
						(f'prev_hth_home_stl_pct_ema{hth_window_size}',   f'post_hth_home_stl_pct_ema{hth_window_size}'),
						(f'prev_hth_home_blk_ema{hth_window_size}',       f'post_hth_home_blk_ema{hth_window_size}'),
						(f'prev_hth_home_blk_pct_ema{hth_window_size}',   f'post_hth_home_blk_pct_ema{hth_window_size}'),
						(f'prev_hth_home_drtg_ema{hth_window_size}',      f'post_hth_home_drtg_ema{hth_window_size}'),
						(f'prev_hth_home_ortg_ema{hth_window_size}',      f'post_hth_home_ortg_ema{hth_window_size}'),
						(f'prev_hth_home_efg_pct_ema{hth_window_size}',   f'post_hth_home_efg_pct_ema{hth_window_size}')]

	ema_a_hth_features = [(f[0].replace('home','away'), f[1].replace('home','away')) for f in ema_h_hth_features]
	sma_h_hth_features = [f[0].replace('ema','sma') for f in ema_h_hth_features]
	sma_a_hth_features = [f.replace('home','away') for f in sma_h_hth_features]




	match_df = pd.read_csv('nba_data.csv')
	match_df['home_win'] = match_df['movl'].map(lambda x: 0 if x < 0 else 1)
	features = ['prev_home_player_elo_avg', 'prev_away_player_elo_avg', 
				'past_5_home_favorite_wins','past_5_away_favorite_wins',
				'home_ml']
	all_ema_features = features + ema_h_features + ema_a_features + ema_h_hth_features +ema_a_hth_features
	all_sma_features = features + sma_h_features + sma_a_features + sma_h_hth_features +sma_a_hth_features

	cutoff = math.floor(len(match_df) * 0.7)
	train = match_df[match_df.index < cutoff].copy()
	test = match_df[match_df.index >= cutoff].copy()

	tscv = TimeSeriesSplit(n_splits=3)
	params = {
        'min_child_weight':[1, 2, 3, 10],
        'alpha':[30,70],
        'gamma':[0.05,0.1,1],
        'lambda':[1,10],
        'subsample':[0.6, 0.8, 1.0],
        'colsample_bytree':[0.6, 0.8, 1.0],
        'max_depth':[6,10,20],
        'n_estimators':[2,3,5,20,50],
        'eta':[0.01,0.1,1]
        }
	xgb_model = xgb.XGBClassifier(tree_method = 'gpu_hist', 
								gpu_id = 0, 
								eval_metric='logloss', 
								random_state = 1,
								use_label_encoder=False)
	X_train = train[all_sma_features]
	X_test = test[all_sma_features]
	y_train = train['home_win']
	y_test = test['home_win']


	grid_search = GridSearchCV(estimator = xgb_model, 
							cv = tscv, 
							scoring = 'accuracy',
							param_grid = params,
							n_jobs = 1,
							verbose=2)
	grid_search.fit(X_train, y_train)
	grid_search.fit(X_train, y_train)
	print(" Results from Grid Search " )
	print("\n The best estimator across ALL searched params:\n",grid_search.best_estimator_)
	print("\n The best score across ALL searched params:\n",grid_search.best_score_)
	print("\n The best parameters across ALL searched params:\n",grid_search.best_params_)
	y_pred = grid_search.best_estimator_.predict(X_test)
	print(accuracy_score(y_test, y_pred))
	file_name = "xgb_sma_tuned.pkl"
	# save
	pickle.dump(grid_search.best_estimator_, open(file_name, "wb"))

if __name__ == "__main__":
    main()
