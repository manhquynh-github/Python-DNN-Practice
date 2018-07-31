# # #
# Make predictions by my_keras_model.h5
# # #

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import keras.utils
import numpy as np

# randomize for initial weights
np.random.seed(7)

# Generate Dataset for making predictions
# This dataset has 3 classes:
# [0]: y = x
# [1]: y = x^2
# [2]: y = x^3
X = []
for i in range(10, 12):
    X.append([i, i, 0])
    X.append([i, i**2, 1])
    X.append([i, i**3, 2])

X = np.array(X)
Y = X[:, 2]
X = X[:, [0,1]]

# # #
# Load model
# # #

model = load_model('my_keras_model.h5')

# # #
# Make predictions
# # #

# calculate predictions
predictions = model.predict(X)

# print result
for i in range(len(X)):
    print("\nX=%s, Y=%s, Predicted=[%s]\nPredict_Prob=%s" % 
          (X[i], Y[i], np.argmax(predictions[i]), predictions[i]))