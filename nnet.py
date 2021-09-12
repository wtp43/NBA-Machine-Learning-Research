
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import pandas as pd
import numpy as np
import time
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE =\
["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

# Hyperparameters
num_epochs = 1000
num_classes = 2
batch_size = 100
learning_rate = 0.001

MODEL_PATH = 'nnet.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


config = {
    "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16])
}


class Net(nn.Module):
	
  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 5)
    self.fc2 = nn.Linear(5, 3)
    self.fc3 = nn.Linear(3, 1)

  def forward(self, x):
    x = nn.functional.relu(self.fc1(x))
    x = nn.functional.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))

def preprocess(data_path):
	window_size = 4
	hth_window_size = 2
	home_features = [(f'prev_home_pts_ema{window_size}',       f'post_home_pts_ema{window_size}'),
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

	away_features = [(f[0].replace('home','away'), f[1].replace('home','away')) for f in home_features]

	home_hth_features = [(f'prev_hth_home_pts_ema{hth_window_size}',       f'post_hth_home_pts_ema{hth_window_size}'),
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

	away_hth_features = [(f[0].replace('home','away'), f[1].replace('home','away')) for f in home_hth_features]

	home_features = [x[0] for x in home_features]
	away_features = [x[0] for x in away_features]
	
	home_hth_features = [x[0] for x in home_hth_features]
	away_hth_features = [x[0] for x in away_hth_features]
	
	df = pd.read_csv(data_path)
	features = ['prev_home_elo', 'prev_away_elo', 
			'home_ml']
	features += home_features + away_features + home_hth_features + away_hth_features
	df = df[features + ['home_win']]
	df = df.dropna(how='any') 
	scaler = MinMaxScaler()
	df[features] = scaler.fit_transform(df[features])

	cutoff = math.floor(len(df) * 0.7)
	train = df[df.index < cutoff].copy()
	test = df[df.index >= cutoff].copy()

	X_train = torch.FloatTensor(train[features].values)
	y_train = torch.FloatTensor(train['home_win'].values)

	X_test = torch.FloatTensor(test[features].values)
	y_test = torch.FloatTensor(test['home_win'].values)

	X_train = X_train.to(device)
	y_train = y_train.to(device)
	X_test = X_test.to(device)
	y_test = y_test.to(device)
	print(X_train.shape, y_train.shape)
	print(X_test.shape, y_test.shape)
	return X_train,y_train,X_test,y_test

def train_cnn(X_train,y_train,X_test,y_test):

	net = Net(X_train.shape[1])
	criterion = nn.BCELoss()
	optimizer = optim.Adam(net.parameters(), lr=learning_rate)

	net = net.to(device)
	criterion = criterion.to(device)

	for epoch in range(num_epochs):
		y_pred = net(X_train)
		y_pred = torch.squeeze(y_pred)
		train_loss = criterion(y_pred, y_train)
		if epoch % 100 == 0:
			train_acc = calculate_accuracy(y_train, y_pred)
			y_test_pred = net(X_test)
			y_test_pred = torch.squeeze(y_test_pred)
			test_loss = criterion(y_test_pred, y_test)
			test_acc = calculate_accuracy(y_test, y_test_pred)
			# print(f'''epoch {epoch}
			# 		Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
			# 		Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
			# 		''')
		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()
	return net


def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)


def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

def evaluate_model(y_test, y_pred, classes = [0,1]):
	y_pred = y_pred.ge(.5).view(-1).cpu()
	y_test = y_test.cpu()
	print(y_pred, y_test)
	print(classification_report(y_test, y_pred, target_names=classes,zero_division=0))
	
	# cm = confusion_matrix(y_test, y_pred)
	# df_cm = pd.DataFrame(cm, index=classes, columns=classes)
	# hmap = sns.heatmap(df_cm, annot=True, fmt="d")
	# hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
	# hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
	# plt.ylabel('True label')
	# plt.xlabel('Predicted label');

def main():
	start_time = time.time()
	print(f'Using {device} device')
	X_train, y_train, X_test, y_test = preprocess('nba_data.csv')
	print(y_test,y_train)
	net = train_cnn(X_train,y_train,X_test,y_test)
	torch.save(net, MODEL_PATH)

	print(f'Test accuracy: {calculate_accuracy(y_test, net(X_test))}')

	print("--- %s seconds ---" % (time.time() - start_time))

	#load nnet
	#net = torch.load(MODEL_PATH)


if __name__ == "__main__":
    main()
