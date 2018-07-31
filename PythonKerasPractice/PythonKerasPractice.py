# # #
# Build simple custom deep neural network with Keras 
# (using Tensorflow 1.8.0 backend)
# Please be sure that both CPU and GPU are used.
# Tested with Intel Core i7 & NVIDIA GeForce 840M
# Result: accuracy = 99.85%
# # #

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import keras.utils
import numpy as np

# randomize for initial weights
np.random.seed(7)

# # #
# Generate Dataset
# # #

# This dataset has 2 classes:
# [0]: y = x
# [1]: y = x^2
X = []
for i in range(1000):
    X.append([i, i, 0])
    X.append([i, i**2, 1])

X = np.array(X)
Y = X[:, 2]
X = X[:, [0,1]]

# # #
# Define Model
# Model: Input (2) -> Hidden Layer 1 (5) [relu] -> Output (2) [softmax]
# # #

model = Sequential()
model.add(Dense(5, input_dim=2, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile model
# loss function = logarithmic loss
# gradient descent algorithm = Adam Stochastic Optimization 
# http://arxiv.org/abs/1412.6980
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# # #
# Fit Model
# # #

# Since output layer has more than one unit,
# the fit process relies on one-hot encoding. 
# Convert labels to categorical one-hot encoding
one_hot_Y = keras.utils.to_categorical(Y, num_classes=2)

# Epochs = number of iterations through dataset
# Batch size = number of instances (data) that are evaluated
# before a weight update
model.fit(X, one_hot_Y, epochs=20, batch_size=10)

# # #
# Save Model
# # #

# save with HDF5 format
model.save('my_keras_model.h5')

# to load the model, use
# model = load_model('my_keras_model.h5')

# # #
# Evaluate Model
# # #

# evaluate using own training dataset
# remember: the evaluate method also relies on one-hot encoding
scores = model.evaluate(X, one_hot_Y)

# print accuracy
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))