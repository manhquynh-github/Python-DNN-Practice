# # #
# Test my_keras_model.h5
# Please be sure that both CPU and GPU are used.
# Tested with Intel Core i7 & NVIDIA GeForce 840M
# Result: accuracy = 100%
# # #

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import keras.utils
import numpy as np

# randomize for initial weights
np.random.seed(7)

# Generate Dataset for testing
# This dataset has 2 classes:
# [0]: y = x
# [1]: y = x^2
X = []
for i in range(1001, 1100):
    X.append([i, i, 0])
    X.append([i, i**2, 1])

X = np.array(X)
Y = X[:, 2]
X = X[:, [0,1]]

# # #
# Load model
# # #

model = load_model('my_keras_model.h5')

# # #
# Test
# # #

# remember: the evaluate method also relies on one-hot encoding
# Convert labels to categorical one-hot encoding
one_hot_Y = keras.utils.to_categorical(Y, num_classes=2)
# evaluate using data test
scores = model.evaluate(X, one_hot_Y)

# print accuracy
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))