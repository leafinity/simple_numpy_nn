import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from net import Net


# date setting
N_SAMPLES = 10000
TEST_SIZE = 0.1


# model setting
lr = 0.01
ep = 1000
bs = 9000


X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
# x = [[-0.3, 0.5], [1.4, -0.4], ...], y = [0, 1, 1, ...]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)


net = Net()
net.add_layer(2, 25, 'relu').add_layer(25, 50, 'relu').add_layer(50, 50, 'relu').add_layer(50, 25, 'relu').add_layer(25, 1, 'sigmoid')
net.train(X_train, y_train, ep, batch_size=bs, learning_rate=lr, optimizer='Adam',
	val_x=X_val, val_y=y_val, early_stop_interval=20, auto_save_interval=10, log=True)