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

def get_feature_probability(yn, sigma, mu):
    p = 1/(np.sqrt(2*np.pi*sigma))*(np.exp((-1*pow(yn-mu, 2))/(2*sigma)))
    return round(p, 5)

def get_evidence(p, y11, y12, y21, y22):
    evidence = p*y11*y12+p*y21*y22
    return evidence

def get_posterior(p, y1, y2, evidence):
    posterior = (p*y1*y2)/evidence
    return posterior