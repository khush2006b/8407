import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X / 255.0
y = to_categorical(y.astype(int))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_base = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model_base.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_base = model_base.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2,
    verbose=0
)

loss_base, acc_base = model_base.evaluate(X_test, y_test, verbose=0)

model_reg = Sequential([
    Dense(256, activation='relu', kernel_regularizer=l2(0.001), input_shape=(784,)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model_reg.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history_reg = model_reg.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2,
    callbacks=[early],
    verbose=0
)

loss_reg, acc_reg = model_reg.evaluate(X_test, y_test, verbose=0)

plt.plot(history_base.history['accuracy'], label='base_train_acc')
plt.plot(history_base.history['val_accuracy'], label='base_val_acc')
plt.plot(history_reg.history['accuracy'], label='reg_train_acc')
plt.plot(history_reg.history['val_accuracy'], label='reg_val_acc')
plt.legend()
plt.title("Accuracy Comparison")
plt.show()

plt.plot(history_base.history['loss'], label='base_train_loss')
plt.plot(history_base.history['val_loss'], label='base_val_loss')
plt.plot(history_reg.history['loss'], label='reg_train_loss')
plt.plot(history_reg.history['val_loss'], label='reg_val_loss')
plt.legend()
plt.title("Loss Comparison")
plt.show()

print("Baseline Test Accuracy:", acc_base)
print("Regularized Test Accuracy:", acc_reg)