import numpy as np
import helpers as h

x1 = []
x2 = []
t = []

with open('data.txt', 'r') as f:
    for line in f.readlines():
        color, roundness, type = line.strip().split(',')
        x1.append(float(color))
        x2.append(float(roundness))
        t.append(int(type))

x1_apples = []
x1_pears = []
x2_apples = []
x2_pears = []

for i in range(len(t)-1):
    x1_apples.append(x1[i]) if t[i] == 1 else x1_pears.append(x1[i])
    x2_apples.append(x2[i]) if t[i] == 1 else x2_pears.append(x2[i])

x1_apples_avg = np.mean(x1_apples)
x2_apples_avg = np.mean(x2_apples)
x1_pears_avg = np.mean(x1_pears)
x2_pears_avg = np.mean(x2_pears)

x1_apples_var = np.var(x1_apples)
x2_apples_var = np.var(x2_apples)
x1_pears_var = np.var(x1_pears)
x2_pears_var = np.var(x2_pears)

p = 0.5

x1_apple_prob = h.get_feature_probability(x1[len(t)-1], x1_apples_var, x1_apples_avg)
x2_apple_prob = h.get_feature_probability(x2[len(t)-1], x2_apples_var, x2_apples_avg)
x1_pear_prob = h.get_feature_probability(x1[len(t)-1], x1_pears_var, x1_pears_avg)
x2_pear_prob = h.get_feature_probability(x2[len(t)-1], x2_pears_var, x2_pears_avg)

evidence = h.get_evidence(p, x1_apple_prob, x2_apple_prob, x1_pear_prob, x2_pear_prob)

posterior_apples = h.get_posterior(p, x1_apple_prob, x2_apple_prob, evidence)
posterior_pears = h.get_posterior(p, x1_pear_prob, x1_pear_prob, evidence)

if posterior_apples > posterior_pears: 
    y =1 
else:
    y = -1

print("Correct") if y == t[len(t)-1] else print("Incorrect")