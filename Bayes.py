

#Need to calculate the posterior fora pple and pear
#and determine which one is greater
#P - prior probability distribution. Assumin 0.5
#p(y|x) - probability by feature, where x is the thing being classified and y is a feature
#p(y|x) = 1/(sqrt(2*pi*sigma^2))*exp((-(yn-mu)^2/2*sigma^2))
#where mu = mean of a feature; sigma^2 = vairance of a feature; yn = feature value for the sample we're trying to classfy
# evidence = P(x1)p(y11|x1)p(y12|x1)+P(x2)p(y21|x2)p(y22|x2)
#so, the posterior for x1 = (P(x1)p(y11|x1)p(y12|x1))/evidence
import numpy as np

def get_feature_probability(yn, sigma, mu):
    p = 1/(np.sqrt(2*np.pi*sigma))*(np.exp((-1*pow(yn-mu, 2))/(2*sigma)))
    return round(p, 5)

def get_evidence(p, y11, y12, y21, y22):
    evidence = p*y11*y12+p*y21*y22
    return evidence

def get_posterior(p, y1, y2, evidence):
    posterior = (p*y1*y2)/evidence
    return posterior

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

x1_apple_prob = get_feature_probability(x1[len(t)-1], x1_apples_var, x1_apples_avg)
x2_apple_prob = get_feature_probability(x2[len(t)-1], x2_apples_var, x2_apples_avg)
x1_pear_prob = get_feature_probability(x1[len(t)-1], x1_pears_var, x1_pears_avg)
x2_pear_prob = get_feature_probability(x2[len(t)-1], x2_pears_var, x2_pears_avg)

evidence = get_evidence(p, x1_apple_prob, x2_apple_prob, x1_pear_prob, x2_pear_prob)

posterior_apples = get_posterior(p, x1_apple_prob, x2_apple_prob, evidence)
posterior_pears = get_posterior(p, x1_pear_prob, x1_pear_prob, evidence)

if posterior_apples > posterior_pears: 
    y =1 
else:
    y = -1

print("Correct") if y == t[len(t)-1] else print("Incorrect")