import numpy as np
import random as rnd

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


x1 = [0.21835, 0.14115, 0.37022, 0.08838, 0.098166]
x2 = [0.81884, 0.83535, 0.8111, 0.62068, 0.79092]
t = [1, 1, 1, -1, -1]
p=0.1

w1 = rnd.random()
w2 = rnd.random()
b = rnd.random()

v = []
e = []
for i in range(5):
    v.append(get_parameter_value(x1[i],w1,x2[i],w2,b))
    e.append(t[i] - get_output(v[i]))

absError = get_error(e)

while absError != 0:
    for i in range(5):
        w1 += get_weight(p, e[i], x1[i])
        w2 += get_weight(p, e[i], x2[i])
        b += get_bias(p, e[i])
    for i in range(5):
        v[i] = get_parameter_value(x1[i],w1,x2[i],w2,b)
        e[i] = t[i] - get_output(v[i])
    absError = get_error(e)
    print(absError)

