import numpy as np

np.random.seed(45)

X = np.random.randn(100, 10)
y = (np.sum(X, axis=1) > 0).astype(int).reshape(-1,1)

noise_idx = np.random.choice(100, 20, replace=False)
X[noise_idx] += np.random.normal(0, 0.5, X[noise_idx].shape)

mask = np.random.rand(*X.shape) < 0.1
X[mask] = 0

split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def init():
    W1 = np.random.randn(10, 64) * 0.01
    b1 = np.zeros((1,64))

    W2 = np.random.randn(64, 32) * 0.01
    b2 = np.zeros((1,32))

    W3 = np.random.randn(32, 1) * 0.01
    b3 = np.zeros((1,1))

    return W1, b1, W2, b2, W3, b3

def forward(X, W1, b1, W2, b2, W3, b3):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    A3 = sigmoid(Z3)

    return Z1, A1, Z2, A2, Z3, A3

def loss(y, y_pred):
    eps = 1e-8
    return -np.mean(y*np.log(y_pred+eps) + (1-y)*np.log(1-y_pred+eps))

def backward(X, y, Z1, A1, Z2, A2, A3, W2, W3):
    m = X.shape[0]

    dZ3 = A3 - y
    dW3 = A2.T @ dZ3 / m
    db3 = np.sum(dZ3, axis=0) / m

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_deriv(Z2)
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0) / m

    return dW1, db1, dW2, db2, dW3, db3

def update(params, grads, lr=0.01):
    for i in range(len(params)):
        params[i] -= lr * grads[i]

params = list(init())

for epoch in range(500):
    Z1, A1, Z2, A2, Z3, A3 = forward(X_train, *params)
    l = loss(y_train, A3)
    grads = backward(X_train, y_train, Z1, A1, Z2, A2, A3, params[2], params[4])
    update(params, grads, lr=0.01)

_, _, _, _, _, y_pred = forward(X_test, *params)
y_pred = (y_pred > 0.5).astype(int)

acc = np.mean(y_pred == y_test)

print("Test Accuracy:", acc)