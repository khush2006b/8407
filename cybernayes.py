# =========================
# CYBERSECURITY IDS (HYBRID NAIVE BAYES)
# =========================

import math
from collections import defaultdict
import numpy as np

# =========================
# SAMPLE DATASET
# =========================
data = [
    ({"device":"mobile","browser":"chrome","failed":3,"freq":20,"duration":120}, "Attack"),
    ({"device":"desktop","browser":"firefox","failed":0,"freq":5,"duration":300}, "Normal"),
    ({"device":"mobile","browser":"chrome","failed":5,"freq":30,"duration":60}, "Attack"),
    ({"device":"desktop","browser":"edge","failed":1,"freq":7,"duration":250}, "Normal"),
    ({"device":"tablet","browser":"chrome","failed":4,"freq":25,"duration":80}, "Attack"),
    ({"device":"desktop","browser":"firefox","failed":0,"freq":6,"duration":280}, "Normal")
]

# =========================
# TRAINING
# =========================
class_counts = defaultdict(int)
cat_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
num_values = defaultdict(lambda: defaultdict(list))
classes = set()

for x, label in data:
    class_counts[label] += 1
    classes.add(label)

    for f,v in x.items():
        if isinstance(v, str):
            cat_counts[label][f][v] += 1
        else:
            num_values[label][f].append(v)

total = len(data)
priors = {c: class_counts[c]/total for c in class_counts}

# mean & variance
mean = defaultdict(dict)
var = defaultdict(dict)

for c in classes:
    for f in num_values[c]:
        vals = num_values[c][f]
        mean[c][f] = np.mean(vals)
        var[c][f] = np.var(vals) + 1e-6

# Gaussian
def gaussian(x, m, v):
    return (1/math.sqrt(2*math.pi*v))*math.exp(-(x-m)**2/(2*v))

# =========================
# PREDICTION
# =========================
def predict(sample):
    scores = {}

    for c in classes:
        logp = math.log(priors[c])

        for f,v in sample.items():
            if isinstance(v, str):
                count = cat_counts[c][f][v]
                total = sum(cat_counts[c][f].values())
                prob = (count+1)/(total+len(cat_counts[c][f]))
                logp += math.log(prob)
            else:
                logp += math.log(gaussian(v, mean[c][f], var[c][f]))

        scores[c] = logp

    return max(scores, key=scores.get)

# =========================
# TEST
# =========================
test_sample = {"device":"mobile","browser":"chrome","failed":4,"freq":28,"duration":90}
print("Prediction:", predict(test_sample))