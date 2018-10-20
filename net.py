import os
import json
import numpy as np

from activation import *
from optimizer import *


class Net(object):
    """ Trainable numpy simple DNN of 2-classes classification"""

    def __init__(self, seed=94):
        super(Net, self).__init__()
        
        np.random.seed(seed)
        
        # for train data
        self.architecture = []
        self.params_values = {}
        self.memory = {}
        self.grads_values = {}
        self.cost_history = []
        self.accuracy_history = []

        # for early_stop
        self.non_progress_steps = 0
        self.maximun_acc = 0.0

        # for auto_save
        self.steps = 0

    def add_layer(self, input_dim, output_dim, activation='relu'):
        ''' Add a layer and initialize weights and biases

        params:

            input_dim: # of previous layer's nodes
            output_dim: # of current layer's node
            activation: activation function name

        weight shape:

            [[W11, W21, W31, ...],]
             [W12, W22, W32, ...],]
             [W13, W23, W33, ...], ...]

            Wij = W[#.node][#.w]

        '''
        self.architecture.append({
            'input_dim': input_dim,
            'output_dim': output_dim,
            'activation': activation
        })
        layer_idx = len(self.architecture)

        # rand`n` -> normal distribution
        # multipling 0.1, because bigger number causes flatter sigmoid
        self.params_values['W' + str(layer_idx)] = np.random.randn(
            input_dim, output_dim) * 0.1

        # 1 bais * # of nodes
        self.params_values['b' + str(layer_idx)] = np.random.randn(
            1, output_dim) * 0.1

        return self

    def load_model(self, filename):
        filename += '.json'
        self.architecture = []

        with open(filename) as f:
            model = json.loads(f.read())
            for i, layer in enumerate(model):
                self.architecture.append({
                    'input_dim':  layer['input_dim'],
                    'output_dim': layer['output_dim'],
                    'activation': layer['activation'],
                })
                self.params_values['W' + str(i+1)] = np.array(model['weights'])
                self.params_values['b' + str(i+1)] = np.array(model['biases'])

    def save_model(self, filename):
        model = []
        for i, layer in enumerate(self.architecture):
            model.append(layer.copy())
            model[-1]['weights'] = self.params_values['W' + str(i+1)].tolist()
            model[-1]['biases'] = self.params_values['b' + str(i+1)].tolist()


        filename += '.json'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(json.dumps(model))

    def train(self, X, Y, epochs, batch_size=0, learning_rate=0.01,
        optimizer='default', val_x=None, val_y=None,
        early_stop_interval=0, auto_save_interval=0, log=False):
        ''' train model with binary crossentropy

            params:
                
                X np.array: [[x11, x12, x13, ...], [x21, x22, x23, ...], ...]
                Y np.array: [y1, y2, y3, ...]
                epochs int: # of epochs
                batch_size int: batch's size, 
                                if not given, batch's size = X'd size
                learning_rate float: learning_rate
                optimizer str: 'dafualt' of 'Adam'
                val_x np.array: [[x11, x12, x13, ...], [x21, x22, x23, ...], ...]
                val_y np.array: [y1, y2, y3, ...]
                early_stop_interval int: stop after n epochs without progress, 
                                        default 0 (no early_stop)
                auto_save_interval int: check accuracy and save model after n epochs,
                                        default 0 (no auto_save)
                log boolean: if True, then log current cost and accuracy 

        '''
        if not batch_size:
            batch_size = X.shape[0]
        batch_steps = int(np.ceil(X.shape[0] / batch_size))

        Y = Y.reshape(Y.shape + (1, ))

        for i in range(epochs):
            for j in range(batch_steps):
                bs_x = X[j * batch_size: (j + 1) * batch_size]
                bs_y = Y[j * batch_size: (j + 1) * batch_size]
                cost, train_acc = self.train_once(bs_x, bs_y, learning_rate, optimizer)
                self.cost_history.append(cost)
                self.accuracy_history.append(train_acc)

            val_acc = 0.0
            if val_x is not None:
                val_y_hat, val_acc = self.predict(val_x, val_y)

            if log:
                print('epochs %d >>> cost: %.3f, acc: %.3f, val_acc: %.3f' % (i, cost, train_acc, val_acc))

            if val_acc:
                acc = val_acc
            else:
                acc = train_acc

            if auto_save_interval:
                self.auto_save(acc, auto_save_interval)

            if early_stop_interval and self.early_stop(acc, early_stop_interval):
                break

            self.maximun_acc = max(self.maximun_acc, acc)

    def predict(self, X, Y=None):
        Y_hat = self.full_forward_propagation(X)
        
        acc = None
        if Y is not None:
            Y = Y.reshape(Y.shape + (1, ))
            acc = Net.get_accurary_value(Y_hat, Y)

        return Net.convert_prob_into_class(Y_hat), acc

    def update(self, learning_rate=0.01, optimizer='default'):
        ''' update weights and biases

            default optimizer:

            W = W - apha * dW
            b = b - apha * db

            apha is learning rate
        '''
        if optimizer == 'default':
            optimizer_func = default_optimizer
        elif optimizer == 'Adam':
            optimizer_func = adam
        else:
            raise Exception('Non-supported optimizer')

        self.params_values = optimizer_func(
            self.params_values, self.grads_values, self.architecture, learning_rate)

    def train_once(self, X, Y, learning_rate=0.01, optimizer='default'):
        Y_hat = self.full_forward_propagation(X)
        cost = self.get_cost_value(Y_hat, Y)
        acc = self.get_accurary_value(Y_hat, Y)

        self.full_backward_propagation(Y_hat, Y)
        self.update(learning_rate, optimizer)

        return cost, acc

    def full_forward_propagation(self, X):
        memory = {}
        A_curr = X

        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1

            self.memory['A' + str(idx)] = A_curr 
            # using inx cause if previous lyaer's output

            activ_function_curr = layer['activation']
            W_curr = self.params_values['W' + str(layer_idx)]
            b_curr = self.params_values['b' + str(layer_idx)]
            A_curr, Z_curr = self.single_layer_forward_propagation(A_curr, W_curr, b_curr, activ_function_curr)

            self.memory['Z' + str(layer_idx)] = Z_curr

        return A_curr

    def full_backward_propagation(self, Y_hat, Y):
        ''' backward propagation

            tho L/ tho Y_hat = - (Y/ Y_hat - (1 - Y)/ (1 - Y_hat))

        '''

        grads_values = {}
        m = Y.shape[1]

        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
        
        for layer_idx_prev, layer in reversed(list(enumerate(self.architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer['activation']

            dA_curr = dA_prev

            A_prev = self.memory['A' + str(layer_idx_prev)] # previous node's output
            Z_curr = self.memory['Z' + str(layer_idx_curr)]
            W_curr = self.params_values['W' + str(layer_idx_curr)]
            b_curr = self.params_values['b' + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            self.grads_values["dW" + str(layer_idx_curr)] = dW_curr
            self.grads_values["db" + str(layer_idx_curr)] = db_curr

    def early_stop(self, acc, steps):
        if acc <= self.maximun_acc:
            self.non_progress_steps += 1
        else:
            self.non_progress_steps = 0
            
        if self.non_progress_steps >= steps:
            return True
            
        return False

    def auto_save(self, acc, steps):
        if self.steps % steps == 0 and acc > self.maximun_acc:
            self.save_model('temp/auto_save_model')

    @staticmethod
    def single_layer_forward_propagation(A_pre, W_curr, b_curr, activation='relu'):
        Z_curr = np.dot(A_pre, W_curr) + b_curr
        if activation == 'relu':
            activation_func = relu
        elif activation == 'sigmoid':
            activation_func = sigmoid
        else:
            raise Exception('Non-supported activation function')

        # return output, output before activation for bp
        return activation_func(Z_curr), Z_curr

    @staticmethod
    def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation='relu'):
        m = A_prev.shape[1]
        if activation == 'relu':
            backward_activation_func = relu_backward
        elif activation == 'sigmoid':
            backward_activation_func = sigmoid_backward
        else:
            raise Exception('Non-supported activation function')
        
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = np.dot(A_prev.transpose(), dZ_curr) / m
        db_curr = np.sum(dZ_curr, axis=0, keepdims=True) / m
        # keepdims sum([[1, 2], [2, 1]], keepdims=True) = [[6]] 
        dA_prev = np.dot(dZ_curr, W_curr.transpose())

        return dA_prev, dW_curr, db_curr

    @staticmethod
    def get_cost_value(Y_hat, Y):
        ''' compute loss for binary crossentropy for 2 class classifiction

            J(W, b) = (1 / m) * sigma i=1~m L(Y_hat_i, Y_i)
            L(Y_hat, Y) = -((Y * log(Y_hat)) + (1 - y)log(1 - Y_hat))

            params:
                Y_hat: prediction
                Y: target

            np.squeeze([[1, 2, 3]])
            >>> [1, 2, 3]
        '''

        m = Y_hat.shape[0]
        a = np.dot(Y.transpose(), np.log(Y_hat))
        b = np.dot(1 - Y.transpose(), 1 - np.log(1 - Y_hat))
        cost = -1 / m * (np.dot(Y.transpose(), np.log(Y_hat)) + np.dot(1 - Y.transpose(), 1 - np.log(1 - Y_hat)))
        return np.squeeze(cost)


    @staticmethod
    def get_accurary_value(Y_hat, Y):
        ''' compute mean square error
            
            params:
                Y_hat: prediction
                Y: target
        '''

        Y_hat = Net.convert_prob_into_class(Y_hat)
        return (Y_hat == Y).all(axis=1).mean()

    @staticmethod
    def convert_prob_into_class(probs):
        ''' find max each row

            # for multi-classification.
            # but the fuction is for 2 classes classifiction
            x = [[1, 2], [4, 3], [5, 1]]
            np.argmax(x, axis=1)
            >>> [1, 0, 0]

        '''
        _probs = np.copy(probs)
        _probs[_probs > 0.5] = 1
        _probs[_probs <= 0.5] = 0
        return _probs



