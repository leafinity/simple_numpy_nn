import numpy as np
import json


_local = {}


def early_stop(steps, acc, interval=0.0001):
    pre_step = _local.setdefault('early_stop_step', 0)
    pre_acc = _local.setdefault('early_stop_acc', 0.0)

    if acc > 0.6 and acc <= pre_acc + interval:
        pre_step += 1
        if pre_step >= steps:
            return True
        _local['early_stop_step'] = pre_step
    else:
        _local['early_stop_acc'] = acc
        _local['early_stop_step'] = 0
        
    return False


def auto_save(params_values, steps, acc, filename):
    cur_step = _local.setdefault('auto_save_step', 0)
    pre_acc = _local.setdefault('auto_save_acc', 0.0)

    if cur_step % steps == 0 and acc > pre_acc:
        save_model(params_values, filename)
        _local['auto_save_acc'] = acc
    
    _local['auto_save_step'] += 1


def save_model(params_values, filename):
    params_values = params_values.copy()
    for k in params_values:
        params_values[k] = params_values[k].tolist()
    with open(filename + '.json', 'w') as f:
        f.write(json.dumps(params_values))


def load_model(filename):
    with open(filename + '.json') as f:
        params_values = json.loads(f.read())

    for k in params_values:
        params_values[k] = np.array(params_values[k])
    return params_values