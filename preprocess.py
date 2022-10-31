import math
import db_func
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
import matplotlib.pyplot as plt
from datetime import date, datetime
from tqdm.notebook import tqdm
import re
from collections import defaultdict


match_df = None
pp_df = None
season_df = None


match_query = '''SELECT
				m.match_id,  m.away_id, m.home_id,
				m.date, m.away_pts, m.home_pts, m.playoff_game,
				h_ml.home_ml, a_ml.away_ml,
				h_ps.home_spread, a_ps.away_spread,
				h_ps.home_ps_odds, a_ps.away_ps_odds,
				over.over, under.under, ou.spread
			FROM match AS m
			LEFT OUTER JOIN
			(
				SELECT
					AVG(decimal_odds) AS home_ml,
					m.match_id AS match_id
				FROM
					odds AS o, team AS t1, team as t2,
					match AS m
				WHERE
					o.bet_type_id = 1 AND
					o.match_id = m.match_id AND
					o.team_id = m.home_id
				GROUP BY m.match_id
			) AS h_ml ON m.match_id = h_ml.match_id
			LEFT OUTER JOIN
			(
				SELECT
					AVG(decimal_odds) AS away_ml,
					m.match_id AS match_id
				FROM
					odds AS o, team AS t1, team as t2,
					match AS m
				WHERE
					o.bet_type_id = 1 AND
					o.match_id = m.match_id AND
					o.team_id = m.away_id
				GROUP BY m.match_id
			) AS a_ml ON m.match_id = a_ml.match_id
			LEFT OUTER JOIN
			(
				SELECT
					AVG(decimal_odds) AS home_ps_odds,
					AVG(spread) AS home_spread,
					m.match_id AS match_id
				FROM
					odds AS o, team AS t1, team as t2,
					match AS m
				WHERE
					o.bet_type_id = 2 AND
					o.match_id = m.match_id AND
					o.team_id = m.home_id
				GROUP BY m.match_id
			) AS h_ps ON m.match_id = h_ps.match_id
			LEFT OUTER JOIN
			(
				SELECT
					AVG(decimal_odds) AS away_ps_odds,
					AVG(spread) AS away_spread,
					m.match_id AS match_id
				FROM
					odds AS o, team AS t1, team as t2,
					match AS m
				WHERE
					o.bet_type_id = 2 AND
					o.match_id = m.match_id AND
					o.team_id = m.away_id
				GROUP BY m.match_id
			) AS a_ps ON m.match_id = a_ps.match_id
			LEFT OUTER JOIN
			(
				SELECT
					AVG(decimal_odds) AS under,
					m.match_id AS match_id
				FROM
					odds AS o, match AS m
				WHERE
					o.bet_type_id = 3 AND
					o.over_under = 'under' AND
					o.match_id = m.match_id
				GROUP BY m.match_id
			) AS under ON m.match_id = under.match_id
			LEFT OUTER JOIN
			(
				SELECT
					AVG(decimal_odds) AS over,
					m.match_id AS match_id
				FROM
					odds AS o, match AS m
				WHERE
					o.bet_type_id = 3 AND
					o.over_under = 'over' AND
					o.match_id = m.match_id
				GROUP BY m.match_id
			) AS over ON m.match_id = over.match_id
			LEFT OUTER JOIN
			(
				SELECT
					AVG(spread) AS spread,
					m.match_id AS match_id
				FROM
					odds AS o, match AS m
				WHERE
					o.bet_type_id = 3 AND
					o.match_id = m.match_id
				GROUP BY m.match_id
			) AS ou ON m.match_id = ou.match_id
			WHERE date >= DATE('2013-10-29')
			ORDER BY date ASC
			'''

season_query = '''SELECT *
				FROM season'''

player_performance_query = '''SELECT p.*, m.date
							FROM player_performance as p, match as m
							WHERE m.match_id = p.match_id
							AND m.date >= DATE('2013-10-29')
							ORDER BY date ASC'''
team_query = '''SELECT * 
				FROM team_name'''

injury_query = '''SELECT i.* 
				FROM injury as i, match as m
				WHERE m.match_id = i.match_id
				AND m.date >= DATE('2013-10-29')
				ORDER BY m.date ASC'''


def get_season(date):
    return season_df[(season_df['start_date'] <= date) &
                     (season_df['end_date'] >= date)]['season'].values[0]


def label_seasons():
	match_df['season'] = match_df['date'].map(get_season)
	pp_df['season'] = pp_df['date'].map(get_season)

def get_prev_match(date, team_id, match_df):
    return match_df[(match_df["date"] < date) &
                    ((match_df["home_id"] == team_id) |
                     (match_df["away_id"] == team_id))].tail(1)

def get_prev_elo(team_id, season, prev_match):
	
    if prev_match.empty:
        prev_elo = 1500.0
    elif team_id == prev_match['home_id'].values[0]:
        prev_elo = prev_match['home_elo'].values[0]
    elif team_id == prev_match['away_id'].values[0]:
        prev_elo = prev_match['away_elo'].values[0]
    else: 
        print('err')

    if (not prev_match.empty and
            (prev_match['season'].values[0]
             != season)):
        prev_elo = prev_elo * 0.75 + 1505 * 0.25
    return prev_elo

def update_elo(home_elo, away_elo, movl):
    elo_diff = home_elo + 100.0 - away_elo
    if movl > 0:
        h_s = 1.0
        a_s = 0.0
        multiplier = ((movl+3)**(0.8))/(7.5+0.006*elo_diff)

    else:
        h_s = 0.0
        a_s = 1.0
        multiplier = ((-movl+3)**(0.8))/(7.5+0.006*(-elo_diff))
        
    exp_h_s = 1.0 / (1.0 + 10.0 ** (-elo_diff/400.0))
    exp_a_s = 1.0 - exp_h_s
    
    k = 20.0 * multiplier

    new_home_elo = home_elo + k * (h_s - exp_h_s)
    new_away_elo = away_elo + k * (a_s - exp_a_s)

    return (new_home_elo, new_away_elo)


def label_elo():
	match_df['home_elo'] = 1500.0
	match_df['away_elo'] = 1500.0
	for idx, row in tqdm(match_df.iterrows(), total=match_df.shape[0]):
		prev_h_match = get_prev_match(row['date'], row['home_id'], match_df)
		prev_a_match = get_prev_match(row['date'], row['away_id'], match_df)
		
		prev_h_elo = get_prev_elo(
			row['home_id'], row['season'], prev_h_match)
		prev_a_elo = get_prev_elo(
			row['away_id'], row['season'], prev_a_match)    
		
		new_elos = update_elo(prev_h_elo, prev_a_elo, row['movl'])
		match_df.at[idx, 'home_elo'] = new_elos[0]
		match_df.at[idx, 'away_elo'] = new_elos[1]
		
		match_df.at[idx, 'prev_home_elo'] = prev_h_elo
		match_df.at[idx, 'prev_away_elo'] = prev_a_elo

def db_to_df():
	conn = db_func.get_conn()
	#insert data from db to df
	match_df = pd.read_sql(match_query, conn)
	season_df = pd.read_sql(season_query, conn)
	pp_df = pd.read_sql(player_performance_query, conn)
	team_df = pd.read_sql(team_query, conn)
	injury_df = pd.read_sql(injury_query, conn)
	match_df['date'] = match_df['date'].map(lambda x: datetime(x.year, x.month, x.day))
	pp_df['date'] = pp_df['date'].map(lambda x: datetime(x.year, x.month, x.day))
	season_df['start_date'] =season_df['start_date'].map(lambda x: datetime(x.year, x.month, x.day))
	season_df['end_date'] = season_df['end_date'].map(lambda x: datetime(x.year, x.month, x.day))
	conn.close()

def label_movl():
	#Add Margin of victory/loss(MOVL) and whether home team won or not
	match_df['movl'] = match_df['home_pts'] - match_df['away_pts']
	match_df['h_win'] = match_df['movl'].map(lambda x: 0 if x < 0 else 1)

def player_match_eff_rating(player):
    per = 0
    if player['sp'] > 0:
        per = player['fg'] * 85.910 
        + player['stl'] * 53.897
        + player['threep'] * 51.757
        + player['ft'] * 46.845
        + player['blk'] * 39.190 
        + player['orb'] * 39.190
        + player['drb'] * 34.677
        + player['ast'] * 14.707
        - player['pf'] * 17.174 
        - (player['fta'] - player['ft']) * 20.091 
        - (player['fga'] - player['fg']) * 39.190
        - player['tov'] * 53.897 
        
        per = per / (player['sp']/60.0)
    return per

def team_match_eff_rating(team_id, match_id, pp_df):
    df = pp_df[(pp_df['team_id'] == team_id) &
                        (pp_df['match_id'] == match_id)]
    return df['per'].sum()

# To get head on head matches for an opponent, set opponent_id
def get_prev_matches(date, team_id, match_df, opponent_id = 0):
    if opponent_id:
        return match_df[(match_df["date"] < date) &
                        (((match_df["home_id"] == team_id) & 
                          (match_df["away_id"] == opponent_id)) |
                         ((match_df["home_id"] == opponent_id) & 
                          (match_df["away_id"] == team_id)))]
    else:
        return match_df[(match_df["date"] < date) &
                    ((match_df["home_id"] == team_id) |
                     (match_df["away_id"] == team_id))]

def get_past_per_sum(team_id, prev_matches, i):
    if len(prev_matches) < i: 
        return None
    prev_matches['res'] =  prev_matches.apply(lambda x:
                             x['home_per'] if x['home_id'] == team_id
                             else x['away_per'], axis=1)
    return prev_matches['res'].sum()    

def get_prev_player_match(date, player_id, pp_df):
    return pp_df[(pp_df['date'] < date) & 
                (pp_df['player_id'] == player_id)].tail(1)
def get_active_players(match_id, team_id, pp_df):
    return  pp_df[(pp_df['match_id'] == match_id) &
                      (pp_df['team_id'] == team_id) &
                  (pp_df['sp']>0)]

def get_complete_roster(match_id, team_id, match_df):
    return  pp_df[(pp_df['match_id'] == match_id) &
                      (pp_df['team_id'] == team_id)]

def label_per():
	pp_df['per'] = pp_df.apply(player_match_eff_rating, axis=1)

	match_df['away_per'] = match_df.apply(lambda x: team_match_eff_rating(
		x['away_id'],x['match_id'], pp_df), axis=1)
	match_df['home_per'] = match_df.apply(lambda x: team_match_eff_rating(
		x['home_id'],x['match_id'], pp_df), axis=1)

	match_df['prev_3_home_per'] = match_df.apply(lambda x: 
									get_past_per_sum(x['home_id'], 
										get_prev_matches(x['date'], 
														x['home_id'],
														match_df).tail(3),
													3), axis=1)
	match_df['prev_3_away_per'] = match_df.apply(lambda x: 
									get_past_per_sum(x['away_id'], 
										get_prev_matches(x['date'], 
														x['away_id'],
														match_df).tail(3),
													3), axis=1)

	match_df['prev_hth_home_per'] = match_df.apply(lambda x: 
									get_past_per_sum(x['home_id'], 
										get_prev_matches(x['date'], 
														x['home_id'],
														match_df,
														x['away_id']
														).tail(1),
													1), axis=1)
	match_df['prev_hth_away_per'] = match_df.apply(lambda x: 
									get_past_per_sum(x['away_id'], 
										get_prev_matches(x['date'], 
														x['away_id'],
														match_df,
														x['home_id']
														).tail(1),
													1), axis=1)

def df_to_db():
	from sqlalchemy import create_engine
	engine = create_engine('postgresql://foka2:#gg3Ewampkl@127.0.0.1:5432/bazakot22')

	df.to_sql("table_name4", engine, if_exists='replace')

def label_team_stats():
	d = defaultdict(list)
	for idx, row in tqdm(match_df.iterrows(), total=match_df.shape[0]):
		home_players = get_active_players(row['match_id'], row['home_id'], pp_df)
		away_players = get_active_players(row['match_id'], row['away_id'], pp_df)
		d['home_bpm'].append(home_players['bpm'].sum())
		d['away_bpm'].append(away_players['bpm'].sum())
		
		d['home_fg'].append(home_players['fg'].sum())
		d['away_fg'].append(away_players['fg'].sum())
		d['home_fg_pct'].append(home_players['fg_pct'].mean())
		d['away_fg_pct'].append(away_players['fg_pct'].mean())
		
		d['home_3p'].append(home_players['threep'].sum())
		d['away_3p'].append(away_players['threep'].sum())
		d['home_3pa'].append(home_players['threepa'].sum())
		d['away_3pa'].append(away_players['threepa'].sum())
		d['home_3p_pct'].append(home_players['threep_pct'].mean())
		d['away_3p_pct'].append(away_players['threep_pct'].mean())
		
		d['home_ft'].append(home_players['ft'].sum())
		d['away_ft'].append(away_players['ft'].sum())
		d['home_ft_pct'].append(home_players['ft_pct'].mean())
		d['away_ft_pct'].append(away_players['ft_pct'].mean())
		
		d['home_orb'].append(home_players['orb'].sum())
		d['away_orb'].append(away_players['orb'].sum())
		d['home_orb_pct'].append(home_players['orb_pct'].mean())
		d['away_orb_pct'].append(away_players['orb_pct'].mean())
		
		d['home_drb'].append(home_players['drb'].sum())
		d['away_drb'].append(away_players['drb'].sum())
		d['home_drb_pct'].append(home_players['drb_pct'].mean())
		d['away_drb_pct'].append(away_players['drb_pct'].mean())
		
		d['home_trb'].append(home_players['trb'].sum())
		d['away_trb'].append(away_players['trb'].sum())
		d['home_trb_pct'].append(home_players['trb_pct'].mean())
		d['away_trb_pct'].append(away_players['trb_pct'].mean())
		
		d['home_tov'].append(home_players['tov'].sum())
		d['away_tov'].append(away_players['tov'].sum())
		d['home_tov_pct'].append(home_players['tov_pct'].mean())
		d['away_tov_pct'].append(away_players['tov_pct'].mean())
		
		d['home_ast'].append(home_players['ast'].sum())
		d['away_ast'].append(away_players['ast'].sum())
		d['home_ast_pct'].append(home_players['ast_pct'].mean())
		d['away_ast_pct'].append(away_players['ast_pct'].mean())
		
		d['home_stl'].append(home_players['stl'].sum())
		d['away_stl'].append(away_players['stl'].sum())
		d['home_stl_pct'].append(home_players['stl_pct'].mean())
		d['away_stl_pct'].append(away_players['stl_pct'].mean())
		
		d['home_blk'].append(home_players['blk'].sum())
		d['away_blk'].append(away_players['blk'].sum())
		d['home_blk_pct'].append(home_players['blk_pct'].mean())
		d['away_blk_pct'].append(away_players['blk_pct'].mean())
		
		d['home_drtg'].append(home_players['drtg'].mean())
		d['away_drtg'].append(away_players['drtg'].mean())
		
		d['home_ortg'].append(home_players['ortg'].mean())
		d['away_ortg'].append(away_players['ortg'].mean())
		
		d['home_efg_pct'].append(home_players['efg_pct'].mean())
		d['away_efg_pct'].append(away_players['efg_pct'].mean())
		
		d['sp'].append(home_players['sp'].sum())
	df = pd.DataFrame(d)
	match_df = pd.concat([match_df.reset_index(drop=True),
						df.reset_index(drop=True)],axis=1)

def ema(current, prev_ema, window_size, smoothing=2.0):
    k = smoothing / (1 + window_size)
    return current * k + prev_ema * (1-k)

def get_prev_team_sum(team_id, home_col, prev_matches):
    away_col = home_col.replace('home', 'away')
    prev_matches['res'] =  prev_matches.apply(lambda x:
                             x[home_col] if x['home_id'] == team_id
                             else x[away_col], axis=1)
    return prev_matches['res'].sum()


def label_sma_ema():
	smoothing = 2

	window_sizes = [3]
	hth_window_sizes = [1,2]

	for w in tqdm(range(len(hth_window_sizes))):
		hth_window_size = hth_window_sizes[w]
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
		sma_h_hth_features = [(f[0].replace('ema','sma'), f[1].replace('ema','sma')) for f in ema_h_hth_features]
		sma_a_hth_features = [(f[0].replace('home','away'), f[1].replace('home','away')) for f in sma_h_hth_features]
		for idx, row in tqdm(match_df.iterrows(), total=match_df.shape[0]):
			prev_hth_matches = get_prev_matches(row['date'], row['home_id'], match_df, row['away_id']).tail(hth_window_size)
			len_prev_hth_matches = len(prev_hth_matches)
			for i in range(len(ema_h_hth_features)):
				h_feature = re.findall('home_.*_ema', ema_h_hth_features[i][0])[0].replace('_ema', '')
				a_feature = h_feature.replace('home', 'away') 
				if not prev_hth_matches.empty:
					prev_match = prev_hth_matches.iloc[-1:]
					match_df.at[idx,sma_h_hth_features[i][0]] = get_prev_team_sum(row['home_id'], h_feature, prev_hth_matches)/len_prev_hth_matches    
					match_df.at[idx,sma_a_hth_features[i][0]] = get_prev_team_sum(row['away_id'], a_feature, prev_hth_matches)/len_prev_hth_matches    


					if len_prev_hth_matches < hth_window_size:
						match_df.at[idx,ema_h_hth_features[i][0]] = match_df.loc[idx,sma_h_hth_features[i][0]] 
						match_df.at[idx,ema_h_hth_features[i][1]] = (match_df.loc[idx,sma_h_hth_features[i][0]] \
																			* len_prev_hth_matches + row[h_feature])/(len_prev_hth_matches + 1)
						match_df.at[idx,ema_a_hth_features[i][0]] = match_df.loc[idx,sma_a_hth_features[i][0]] 
						match_df.at[idx,ema_a_hth_features[i][1]] = (match_df.loc[idx,sma_a_hth_features[i][0]] \
																			* len_prev_hth_matches + row[a_feature])/(len_prev_hth_matches + 1)
					else:
						match_df.at[idx,ema_h_hth_features[i][0]] = prev_match[ema_h_hth_features[i][1]] \
																if prev_match['home_id'].values[0] == row['home_id'] \
																else prev_match[ema_a_hth_features[i][1]]

						match_df.at[idx,ema_h_hth_features[i][1]] = ema(row[h_feature],  
																	match_df.loc[idx,ema_h_hth_features[i][0]], 
																	hth_window_size)
						match_df.at[idx,ema_a_hth_features[i][0]] = prev_match[ema_h_hth_features[i][1]] \
																if prev_match['home_id'].values[0] == row['home_id'] \
																else prev_match[ema_a_hth_features[i][1]]

						match_df.at[idx,ema_a_hth_features[i][1]] = ema(row[h_feature],  
																	match_df.loc[idx,ema_a_hth_features[i][0]], 
																	hth_window_size)
				else:
					match_df.at[idx,ema_h_hth_features[i][1]] = row[h_feature]
					match_df.at[idx,ema_a_hth_features[i][1]] = row[a_feature]


	for w in tqdm(range(len(window_sizes))):
		window_size = window_sizes[w]

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
		sma_h_features = [(f[0].replace('ema','sma'), f[1].replace('ema','sma')) for f in ema_h_features]
		sma_a_features = [(f[0].replace('home','away'), f[1].replace('home','away')) for f in sma_h_features]

		for idx, row in tqdm(match_df.iterrows(), total=match_df.shape[0]):
			prev_h_matches = get_prev_matches(row['date'], row['home_id'], match_df).tail(window_size)
			prev_a_matches = get_prev_matches(row['date'], row['away_id'], match_df).tail(window_size)
			len_prev_h_matches = len(prev_h_matches)
			len_prev_a_matches = len(prev_a_matches)
			for i in range(len(ema_h_features)):
				h_feature = re.findall('home_.*_ema', ema_h_features[i][0])[0].replace('_ema', '')
				a_feature = h_feature.replace('home', 'away') 

				if not prev_h_matches.empty:
					prev_match = prev_h_matches.iloc[-1:]
					match_df.at[idx,sma_h_features[i][0]] = get_prev_team_sum(row['home_id'], h_feature, prev_h_matches)/len_prev_h_matches    

					if len_prev_h_matches < window_size:
						match_df.at[idx,ema_h_features[i][0]] = match_df.loc[idx,sma_h_features[i][0]] 
						match_df.at[idx,ema_h_features[i][1]] = (match_df.loc[idx,sma_h_features[i][0]] \
																			* len_prev_h_matches + row[h_feature])/(len_prev_h_matches + 1)
					else:
						match_df.at[idx,ema_h_features[i][0]] = prev_match[ema_h_features[i][1]] \
																if prev_match['home_id'].values[0] == row['home_id'] \
																else prev_match[ema_a_features[i][1]]

						match_df.at[idx,ema_h_features[i][1]] = ema(row[h_feature],  
																	match_df.loc[idx,ema_h_features[i][0]], 
																	window_size)
				else:
					match_df.at[idx,ema_h_features[i][1]] = row[h_feature]


				if not prev_a_matches.empty:
					prev_match = prev_a_matches.iloc[-1:]
					match_df.at[idx,sma_a_features[i][0]] = get_prev_team_sum(row['away_id'], h_feature, prev_a_matches)/len_prev_a_matches

					if len_prev_a_matches < window_size:
						match_df.at[idx,ema_a_features[i][0]] = match_df.loc[idx,sma_a_features[i][0]] 
						match_df.at[idx,ema_a_features[i][1]] = (match_df.loc[idx,sma_a_features[i][0]] \
																			* len_prev_a_matches + row[a_feature])/(len_prev_a_matches + 1)
					else:
						match_df.at[idx,ema_a_features[i][0]] = prev_match[ema_h_features[i][1]] \
																if prev_match['home_id'].values[0] == row['home_id'] \
																else prev_match[ema_a_features[i][1]]

						match_df.at[idx,ema_a_features[i][1]] = ema(row[a_feature],  
																	match_df.loc[idx,ema_a_features[i][0]], 
																	window_size)
				else:
					match_df.at[idx,ema_a_features[i][1]] = row[a_feature]


def get_past_wins_as_favorite(team_id, prev_matches, i):
    if len(prev_matches) < i: 
        return None
    prev_matches['res'] =  prev_matches.apply(lambda x:
                             1 if (x['home_id'] == team_id and x['favorite'] and x['favorite_won']) or 
                                      (x['away_id'] == team_id and not x['favorite'] and x['favorite_won'])        
                             else 0, axis=1)
    return prev_matches['res'].sum()/i    

def label_consistency():
	# favorite = 1: home team is favorite.
	# favorite = 0: away team is favorite
	match_df['favorite'] = match_df['home_ml'] < match_df['away_ml']
	match_df['favorite_won'] = match_df.apply(lambda x: (x['favorite'] and x['h_win'] == 1) or
											(not x['favorite'] and x['h_win'] == 0), axis=1)
	window_sizes = [2,3,4,5,6,7,8,9,10]

	for w in tqdm(window_sizes):
		match_df[f'past_{w}_home_favorite_wins'] = match_df.apply(lambda x: 
										get_past_wins_as_favorite(x['home_id'], 
											get_prev_matches(x['date'], 
															x['home_id'],
															match_df
															).tail(w),
														w), axis=1)
		match_df[f'past_{w}_away_favorite_wins'] = match_df.apply(lambda x: 
										get_past_wins_as_favorite(x['away_id'], 
											get_prev_matches(x['date'], 
															x['away_id'],
															match_df
															).tail(w),
														w), axis=1)

def df_to_csv():
	season_df.to_csv('seasons.csv', header=True, index=False)
	match_df.to_csv('matches.csv', header=True, index=False)
	pp_df.to_csv('playerperformances.csv', header=True, index=False)
	injury_df.to_csv('injuries.csv', header=True, index=False)


def main():
	#Fill dataframes
	db_to_df()

	#Define seasons
	label_seasons()


	#Filter rows based on when 3 point era
	label_movl()

	#Player EFficiency Rating over a sum of the last 3 games
	label_per()


	#SMA and EMA stats
	label_sma_ema()

	#Team Consistency
	label_consistency()

	#Team Elo Rating
	label_elo()

	#Save df to csv
	df_to_csv()

if __name__ == '__main__':
	main()