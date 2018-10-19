import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    ''' relu(Z)

        if x >= 0, f(x) = x
        if x <  0, f(x) = 0
    '''
    return np.maximum(0, Z)

def sigmoid_backward(dA, Z):
    ''' sigmoid'(Z)

        dA: current layer's output
        Z: output before activ_func

        f(x) = e^x  ->  f'(x) = e^x

        f(g(h(i(x)) = f'(g(x)) * g'(h(x)) * h'(i(x)) * i'(x)
        f(x) = 1 / x  ->  f'(x) = - 1 / x ^ 2
        g(x) = 1 + x  ->  g'(x) = 1
        h(x) = e^x = e^x
        i(x) = -x

        f(x) = 1 / (1 + e^(-x))
        f'(x) = (-1 / (1 + e^(-x)) ^ 2) * 1 * e^(-x) * -1
              = (1 / (1 + e^(-x)) * (e^(-x) / (1 + e^(-x))
              = sigmoid(x) * ((1 + e^(-x) - 1)/ (1 + e^(-x)))
              = sigmoid(x) * (1 - sigmoid(x))
    ''' 

    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    ''' relu'(Z)

        Z > 0, relu' = 1
        Z < 0, relu' = 0

        dA: current layer's output
        Z: output before activ_func
    '''

    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    return dZ
