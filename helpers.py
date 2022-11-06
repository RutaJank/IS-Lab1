import numpy as np


def get_example_error(desired_output, y):
    error = desired_output - y
    return error

def get_error(errors):
    error = sum(np.absolute(errors))
    return error

def get_parameter_value(x1, w1, x2, w2, b):
    v = x1*w1 + x2*w2 + b
    return v

def get_bias(p, error):
    bias = p * error
    return bias

def get_weight(p ,e, x):
    w = p*e*x
    return w

def get_output(v):
    if v > 0:
        y = 1
    else:
        y = -1
    return y