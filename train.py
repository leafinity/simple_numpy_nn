import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from nn import train, predict


# date setting
N_SAMPLES = 10000
TEST_SIZE = 0.1

# model setting
lr = 0.01
ep = 10000
bs = 64
nn_architecture = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]


X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
# x = [[-0.3, 0.5], [1.4, -0.4], ...], y = [0, 1, 1, ...]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)



params_values, cost_history, accuracy_history = train(X_train, y_train, nn_architecture, 
    epochs=ep, batch_size=bs, learning_rate=lr, optimizer='Adam',
    load_model_filename='temp', early_stop_step=20, auto_save_filename='temp',
    val_x=X_val, val_y=y_val, log=True)

Y_predict, acc = predict(X_test, y_test, params_values, nn_architecture)
print(acc)