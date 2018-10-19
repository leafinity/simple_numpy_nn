import numpy as np

_local = {}

def default_optimizer(params_values, grads_values, nn_architecture, lr=0.01):
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        params_values['W' + str(layer_idx)] -= lr * grads_values['dW' + str(layer_idx)]
        params_values['b' + str(layer_idx)] -= lr * grads_values['db' + str(layer_idx)]

    return params_values


def adam(params_values, grads_values, nn_architecture, alpha=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-3):
    ''' Adam Optimizer: RMSProp + Momentum

        alpha: step size
        beta_1, beta_2: belong to [0, 1), 
                        Exponential decay rates for the moment estimates

    '''
    # get cache, or initialize time step, 1st, 2nd moment vector
    t = _local.get('t', 0)

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1

        # update weight
        for i in ['W', 'b']:
            m = _local.get('m' + i + str(layer_idx), 0)
            v = _local.get('v' + i + str(layer_idx), 0)
            gt = grads_values['d' + i + str(layer_idx)]
    
            t += 1
            # Update biased first moment estimate
            m = beta_1 * m + (1 - beta_1) * gt
            # Update biased second raw moment estimate
            v = beta_2 * v + (1 - beta_2) * np.square(gt)
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta_1 ** t)
            # Compute bias-corrected second raw moment estimate

            v_hat = v / (1 - beta_2 ** t)
            # Update parameters
            params_values[i + str(layer_idx)] -= (alpha * m_hat) / (np.sqrt(t) + epsilon)

            # save cache
            _local['m' + i + str(layer_idx)] = m
            _local['v' + i + str(layer_idx)] = v

    _local['t'] = t

    return params_values