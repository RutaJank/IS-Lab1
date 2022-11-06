import helpers as h
import random as rnd

x1, x2, t = h.get_data()

p=0.1

w1 = rnd.random()
w2 = rnd.random()
b = rnd.random()

v = []
e = []
for i in range(5):
    v.append(h.get_parameter_value(x1[i],w1,x2[i],w2,b))
    e.append(t[i] - h.get_output(v[i]))

absError = h.get_error(e)

while absError != 0:
    for i in range(5):
        w1 += h.get_weight(p, e[i], x1[i])
        w2 += h.get_weight(p, e[i], x2[i])
        b += h.get_bias(p, e[i])
    for i in range(5):
        v[i] = h.get_parameter_value(x1[i],w1,x2[i],w2,b)
        e[i] = t[i] - h.get_output(v[i])
    absError = h.get_error(e)
    print(absError)

#Iterate over whole dataset and check if classified correctly
for i in range(len(t)):
    vv = h.get_parameter_value(x1[i],w1,x2[i],w2,b)
    y = h.get_output(vv)
    print("Correct") if y == t[i] else print("Incorrect")
