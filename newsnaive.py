# =========================
# FAKE NEWS DETECTION
# =========================

import math
from collections import defaultdict
import re

# =========================
# DATASET
# =========================
data = [
    ("Breaking news government collapse", "Fake"),
    ("Scientists discover new medicine", "Real"),
    ("Shocking celebrity scandal revealed", "Fake"),
    ("Economy growth improves steadily", "Real"),
    ("Fake news spreading rapidly online", "Fake"),
    ("New research shows climate change impact", "Real")
]

stopwords = {"the","is","and","in","of","to","a"}

# =========================
# PREPROCESS
# =========================
def preprocess(text):
    words = re.findall(r'\w+', text.lower())
    return [w for w in words if w not in stopwords]

# =========================
# TRAINING
# =========================
class_counts = defaultdict(int)
word_counts = defaultdict(lambda: defaultdict(int))
total_words = defaultdict(int)
vocab = set()

for text,label in data:
    class_counts[label] += 1
    words = preprocess(text)

    for w in words:
        word_counts[label][w] += 1
        total_words[label] += 1
        vocab.add(w)

total_docs = len(data)
priors = {c: class_counts[c]/total_docs for c in class_counts}

# =========================
# PREDICT
# =========================
def predict(text):
    words = preprocess(text)
    scores = {}

    for c in class_counts:
        logp = math.log(priors[c])

        for w in words:
            count = word_counts[c][w]
            prob = (count+1)/(total_words[c]+len(vocab))
            logp += math.log(prob)

        scores[c] = logp

    return max(scores, key=scores.get)

# =========================
# TEST
# =========================
test_news = "government announces shocking scandal"
print("Prediction:", predict(test_news))