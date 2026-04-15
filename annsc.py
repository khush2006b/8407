# =========================
# ANN FROM SCRATCH WITH REAL DATASET
# =========================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)

# =========================
# LOAD DATASET
# =========================
data = fetch_california_housing()

X = data.data
y = data.target.reshape(-1,1)

# Select 5 features (as required in assignment)
X = X[:, :5]

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# ACTIVATIONS
# =========================
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# =========================
# INITIALIZATION
# =========================
def init():
    W1 = np.random.randn(5, 16) * 0.01
    b1 = np.zeros((1,16))

    W2 = np.random.randn(16, 8) * 0.01
    b2 = np.zeros((1,8))

    W3 = np.random.randn(8, 1) * 0.01
    b3 = np.zeros((1,1))

    return W1, b1, W2, b2, W3, b3

# =========================
# FORWARD
# =========================
def forward(X, W1, b1, W2, b2, W3, b3):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    A3 = Z3  # linear

    return Z1, A1, Z2, A2, Z3, A3

# =========================
# LOSS
# =========================
def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

# =========================
# BACKPROP
# =========================
def backward(X, y, Z1, A1, Z2, A2, A3, W2, W3):
    m = X.shape[0]

    dZ3 = 2*(A3 - y)
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

# =========================
# UPDATE
# =========================
def update(params, grads, lr=0.01):
    for i in range(len(params)):
        params[i] -= lr * grads[i]

# =========================
# TRAINING
# =========================
params = list(init())

losses = []

for epoch in range(500):
    Z1, A1, Z2, A2, Z3, A3 = forward(X_train, *params)

    loss = mse(y_train, A3)
    losses.append(loss)

    grads = backward(X_train, y_train, Z1, A1, Z2, A2, A3, params[2], params[4])

    update(params, grads, lr=0.01)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# =========================
# TEST
# =========================
_, _, _, _, _, y_pred = forward(X_test, *params)

print("\nTest MSE:", mse(y_test, y_pred))

# =========================
# PLOT LOSS
# =========================
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

# =========================
# ACTUAL VS PREDICTED
# =========================
plt.scatter(y_test[:100], y_pred[:100])
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()

# =========================
# NEW PREDICTION
# =========================
new_house = X_test[0].reshape(1,-1)
_, _, _, _, _, pred = forward(new_house, *params)

print("\nSample Prediction:", pred[0][0])