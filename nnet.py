
import torch
from torch import nn, optim
import torch.nn.functional as F
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
import os
from sklearn.metrics import accuracy_score

import optuna
from optuna.trial import TrialState

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
EPOCHS = 2000
CLASSES = 2
LEARNING_RATE = 0.001

MODEL_PATH = 'nnet.pytorch'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIR = os.getcwd()
LOG_INTERVAL = 10


class Net(nn.Module):
	
  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 20)
    self.fc2 = nn.Linear(20, 10)
    self.fc3 = nn.Linear(10, 1)
	# for i in range(0, num_hidden_layers):
	# 	self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

  def forward(self, x):
    x = nn.functional.relu(self.fc1(x))
    x = nn.functional.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))



# config = {
#     "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#     "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#     "lr": tune.loguniform(1e-4, 1e-1),
#     "batch_size": tune.choice([2, 4, 8, 16])
# }

def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 10)
    layers = []

    in_features = len(get_features())
    for i in tqdm(range(n_layers)):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

def objective(trial):

	# Generate the model.
	model = define_model(trial).to(DEVICE)
	train, val, test = preprocess('nba_data.csv')

	# Generate the optimizers.
	optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
	lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
	optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

	features = get_features()
	X_train = torch.FloatTensor(train[features].values)
	y_train = torch.tensor(train['home_win'].values)

	X_val = torch.FloatTensor(val[features].values)
	y_val = torch.tensor(val['home_win'].values)

	X_test = torch.FloatTensor(test[features].values)
	y_test = torch.tensor(test['home_win'].values)

	X_train = X_train.to(DEVICE)
	y_train = y_train.to(DEVICE)
	X_val = X_val.to(DEVICE)
	y_val = y_val.to(DEVICE)
	X_test = X_test.to(DEVICE)
	y_test = y_test.to(DEVICE)

	# Training of the model.
	for epoch in tqdm(range(EPOCHS)):
		model.train()

		optimizer.zero_grad()
		output = model(X_train)
		loss = F.nll_loss(output, y_train)
		loss.backward()
		optimizer.step()

		# Validation of the model.
		model.eval()

		with torch.no_grad():
			output = model(X_val)
			# Get the index of the max log-probability.
			pred = output.argmax(dim=1, keepdim=True)
			correct = pred.eq(y_val.view_as(pred)).sum().item()

		accuracy = correct / len(pred)

		trial.report(accuracy, epoch)

		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()
	return accuracy


def get_features():
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
	features = ['prev_home_elo', 'prev_away_elo', 
			'home_ml']
	features += home_features + away_features + home_hth_features + away_hth_features
	return features

def preprocess(data_path):
	features = get_features()
	df = pd.read_csv(data_path)
	df = df[features + ['home_win']]
	df = df.dropna(how='any') 
	scaler = MinMaxScaler()
	df[features] = scaler.fit_transform(df[features])

	train_cutoff = math.floor(len(df) * 0.6)
	val_cutoff = math.floor(len(df) * 0.8)
	train = df[df.index < train_cutoff]

	val = df[(df.index >= train_cutoff) & 
			(df.index < val_cutoff)]
	test = df[df.index >= val_cutoff]
	return train, val, test


def train_cnn(train, val, test):
	features = get_features()
	X_train = torch.FloatTensor(train[features].values)
	y_train = torch.FloatTensor(train['home_win'].values)

	X_val = torch.FloatTensor(val[features].values)
	y_val = torch.FloatTensor(val['home_win'].values)

	X_test = torch.FloatTensor(test[features].values)
	y_test = torch.FloatTensor(test['home_win'].values)

	X_train = X_train.to(DEVICE)
	y_train = y_train.to(DEVICE)
	X_val = X_val.to(DEVICE)
	y_val = y_val.to(DEVICE)
	X_test = X_test.to(DEVICE)
	y_test = y_test.to(DEVICE)
	print(X_train.shape, y_train.shape)
	print(X_val.shape, y_val.shape)
	print(X_test.shape, y_test.shape)


	net = Net(X_train.shape[1])
	criterion = nn.BCELoss()
	optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

	net = net.to(DEVICE)
	criterion = criterion.to(DEVICE)

	for epoch in range(EPOCHS):
		y_pred = net.forward(X_train)
		y_pred = torch.squeeze(y_pred)
		train_loss = criterion(y_pred, y_train)
		if epoch % 100 == 0:
			train_acc = calculate_accuracy(y_train, y_pred)
			y_val_pred = net(X_val)
			y_val_pred = torch.squeeze(y_val_pred)
			test_loss = criterion(y_val_pred, y_val)
			test_acc = calculate_accuracy(y_val, y_val_pred)
			print(f'''epoch {epoch}
					Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
					Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
					''')
		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()
	print(f'Test accuracy: {calculate_accuracy(y_test, net(X_test))}')
	return net


def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)


def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

def evaluate_model(y_test, y_pred, classes = ['0','1']):
	y_pred = y_pred.ge(.5).view(-1).cpu()
	y_test = y_test.cpu()
	print(y_pred, y_test)
	print(classification_report(y_test, y_pred, target_names=classes))
	
	# cm = confusion_matrix(y_test, y_pred)
	# df_cm = pd.DataFrame(cm, index=classes, columns=classes)
	# hmap = sns.heatmap(df_cm, annot=True, fmt="d")
	# hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
	# hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
	# plt.ylabel('True label')
	# plt.xlabel('Predicted label');

def plot_history(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Loss and Accuracy');	

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def main():
	start_time = time.time()
	print(f'Using {DEVICE} device')
	train, val, test = preprocess('nba_data.csv')
	net = train_cnn(train, val, test)
	torch.save(net, MODEL_PATH)
	
	features = get_features()
	X_test = torch.FloatTensor(test[features].values)
	y_test = torch.FloatTensor(test['home_win'].values)
	y_pred = net.forward(X_test)
	y_pred = torch.squeeze(y_pred)

	evaluate_model(y_test, y_pred)
	#print(f'Test accuracy: {calculate_accuracy(y_test, net(X_test))}')

	print("\n--- %s seconds ---" % (time.time() - start_time))

	#load nnet
	#net = torch.load(MODEL_PATH)


def optuna_optimize():
	start_time = time.time()
	print(f'Using {DEVICE} device')

	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=1000)
	pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
	complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))

	train, val, test = preprocess('nba_data.csv')

	features = get_features()
	X_test = torch.FloatTensor(test[features].values)
	y_test = torch.tensor(test['home_win'].values)

	X_test = X_test.to(DEVICE)
	y_test = y_test.to(DEVICE)

	best_model = define_model(study.best_trial).to(DEVICE)
	y_pred = best_model(X_test)
	y_pred = y_pred.argmax(dim=1, keepdim=True)
	print(y_pred, y_test)
	print(y_pred.eq(y_test.view_as(y_pred)))
	with torch.no_grad():
		evaluate_model(y_test, y_pred)
		correct = y_pred.eq(y_test.view_as(y_pred)).sum().item()
	print(f'Test Accuracy: {correct / len(y_test)}')
	print("\n--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    optuna_optimize()
