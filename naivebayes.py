from collections import defaultdict
import math
import random

data = [
 (["free","money","now"], "spam"),
 (["limited","offer","money"], "spam"),
 (["hello","how","are","you"], "ham"),
 (["congratulations","you","win"], "spam"),
 (["meet","me","for","lunch"], "ham"),
 (["free","offer","money"], "spam"),
 (["hello","dear","friend"], "ham"),
 (["money","limited","free"], "spam"),
 (["how","are","you","today"], "ham")
]

# ---- Split ----
random.shuffle(data)
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]

# ---- Train ----
def train(data):
    wc = {"spam":defaultdict(int), "ham":defaultdict(int)}
    cc = {"spam":0, "ham":0}
    vocab = set()
    
    for words, label in data:
        cc[label] += 1
        for w in words:
            wc[label][w] += 1
            vocab.add(w)
    
    return wc, cc, vocab

# ---- Predict ----
def predict(words, wc, cc, vocab):
    total = cc["spam"] + cc["ham"]
    V = len(vocab)
    
    scores = {}
    for c in ["spam","ham"]:
        logp = math.log(cc[c]/total)
        total_words = sum(wc[c].values())
        
        for w in words:
            logp += math.log((wc[c][w] + 1) / (total_words + V))
        
        scores[c] = logp
    
    return max(scores, key=scores.get)
0.
# # ---- Evaluate ----
# wc, cc, vocab = train(train_data)

# correct = 0
# for words, label in test_data:
#     if predict(words, wc, cc, vocab) == label:
#         correct += 1

# accuracy = correct / len(test_data)
# print("Accuracy:", accuracy)

# ---- User Input ----
wc, cc, vocab = train(train_data)
user_input = input("Enter email: ")

# convert sentence → list of words
words = user_input.lower().split()

# predict
result = predict(words, wc, cc, vocab)

print("Prediction:", result)