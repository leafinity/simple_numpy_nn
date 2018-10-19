import numpy as np

from activation import *
from optimizer import *
from utils import *


def init_layers(nn_architecture, seed=94):
    ''' initialize weight of each layer

        params:
            list nn_architecture: list of layers' architecture
                dict architecture of each layer: 
                {
                    input_dim: # of previous layer's nodes
                    output_dim: # of current layer's node
                    activation: activation function name
                }

    '''
    np.random.seed(seed)
    numbers_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1  # starting from 1
        layer_input_size = layer['input_dim']       
        layer_output_size = layer['output_dim']     

        params_values['W' + str(layer_idx)] = np.random.randn( # rand`n` -> normal distribution
            layer_output_size, layer_input_size) * 0.1 
            # each node has layer_input_size weight
            # total layer_output_size nodes

            # multipling 0.1 cause bigger number cause flatter sigmoid

        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
            # each node has 1 bais
            # total layer_output_size nodes

    return params_values


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
    cost = -1 / m * (np.dot(Y.transpose(), np.log(Y_hat)) + np.dot(1 - Y.transpose(), 1 - np.log(1 - Y_hat)))
    return np.squeeze(cost)


def get_accurary_value(Y_hat, Y):
    ''' compute mean square error
        
        params:
            Y_hat: prediction
            Y: target
    '''

    Y_hat = convert_prob_into_class(Y_hat)
    return (Y_hat == Y).all(axis=1).mean()

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


def single_layer_forward_propagation(A_pre, W_curr, b_curr, activation='relu'):
    Z_curr = np.dot(A_pre, W_curr.transpose()) + b_curr.transpose()
    if activation == 'relu':
        activation_func = relu
    elif activation == 'sigmoid':
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')

    # return output, Z of backward
    return activation_func(Z_curr), Z_curr


def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1

        memory['A' + str(idx)] = A_curr 
        # using inx cause if previous lyaer's output

        activ_function_curr = layer['activation']
        W_curr = params_values['W' + str(layer_idx)]
        b_curr = params_values['b' + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_curr, W_curr, b_curr, activ_function_curr)

        memory['Z' + str(layer_idx)] = Z_curr

    return A_curr, memory


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation='relu'):
    m = A_prev.shape[1]

    if activation == 'relu':
        backward_activation_func = relu_backward
    elif activation == 'sigmoid':
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr.transpose(), A_prev) / m
    db_curr = np.sum(dZ_curr.transpose(), axis=1, keepdims=True)
    # keepdims sum([[1, 2], [2, 1]], keepdims=True) = [[6]] 
    dA_prev = np.dot(dZ_curr, W_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    ''' full backward propagation

        tho L/ tho Y_hat = - (Y/ Y_hat - (1 - Y)/ (1 - Y_hat))

    '''

    grads_values = {}
    m = Y.shape[1]

    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer['activation']

        dA_curr = dA_prev

        A_prev = memory['A' + str(layer_idx_prev)] # previous node's output
        Z_curr = memory['Z' + str(layer_idx_curr)]
        W_curr = params_values['W' + str(layer_idx_curr)]
        b_curr = params_values['b' + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate, optimizer='default'):
    ''' update weights and biases

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

    return optimizer_func(params_values, grads_values, nn_architecture, learning_rate)

def train_once(X, Y, params_values, nn_architecture, learning_rate, optimizer='default'):
    Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
    cost = get_cost_value(Y_hat, Y)
    acc = get_accurary_value(Y_hat, Y)

    grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
    params_values = update(params_values, grads_values, nn_architecture, learning_rate, optimizer)

    return params_values, cost, acc


def train(X, Y, nn_architecture, epochs, batch_size, learning_rate,
    optimizer='default', val_x=None, val_y=None, load_model_filename='',
    early_stop_step=0, auto_save_filename='', log=False):

    params_values = init_layers(nn_architecture)
    batch_steps = X.shape[0] // batch_size
    Y = Y.reshape(Y.shape + (1, ))
    cost_history = []
    accuracy_history = []

    if load_model_filename:
        model = load_model(load_model_filename)
        for k in params_values:
            params_values[k] = model[k]

    for i in range(epochs):
        # if epochs % 10 == 0:
        #     learning_rate *= 1e-1
        for j in range(batch_steps):
            bs_x = X[j * batch_size: (j + 1) * batch_size]
            bs_y = Y[j * batch_size: (j + 1) * batch_size]
            params_values, cost, train_acc = train_once(bs_x, bs_y,
                params_values, nn_architecture, learning_rate, optimizer)
            cost_history.append(cost)
            accuracy_history.append(train_acc)

        val_acc = 0.0
        if val_x is not None:
            val_y_hat, val_acc = predict(val_x, val_y, params_values, nn_architecture)

        if log:
            print('epochs %d >>> cost: %.3f, acc: %.3f, val_acc: %.3f' % (i, cost, train_acc, val_acc))

        if val_acc:
            acc = val_acc
        else:
            acc = train_acc

        if auto_save_filename:
            auto_save(params_values, 5, acc, auto_save_filename)

        if early_stop_step and early_stop(early_stop_step, acc):
            break


    return params_values, cost_history, accuracy_history

def predict(X, Y, params_values, nn_architecture):
    Y = Y.reshape(Y.shape + (1, ))
    Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
    
    acc = None
    if Y is not None:
        acc = get_accurary_value(Y_hat, Y)

    return Y_hat, acc
