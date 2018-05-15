"""
https://www.youtube.com/watch?v=yX8KuPZCAMo
https://www.datacamp.com/community/tutorials/tensorflow-tutorial
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


# reading the dataset
def read_dataset():
    df = pd.read_csv("sonar.csv")

    # features
    X = df[df.columns[0:60]].values
    y = df[df.columns[60]]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    print(X.shape)
    return X, Y


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# =================
# define the input data and the parameters
# =================

X, Y = read_dataset()
X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.10, random_state=415)

learning_rate = 0.01
training_epochs = 2500
n_dim = X.shape[1]  # number of features (number of columns)
n_class = 2  # number of labels
model_path = dir_path+"\\out_model"

# number of hidden layers and units for each hidden layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])  # none means that can be any value
y_ = tf.placeholder(tf.float32, [None, n_class])

# history variables used to visualize the error, accuracy and cost of each epoch
mse_history = []
accuracy_history = []
cost_history = []


def multilayer_perceptron(x, weights, biases):
    # hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])  # matmul = matrix multiplication
    layer_1 = tf.nn.sigmoid(layer_1)

    # hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # hidden layer with sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # hidden layer with ReLU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


# define the weights and the biases for each layer
weights = {
    # n_dim,n_hidden_1 indicates the shape of this particular tensor
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

# =================
# define the functions
# =================
init_function = tf.global_variables_initializer()

nn_function = multilayer_perceptron(x, weights, biases)
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_function, labels=y_))  # loss
training_function = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

prediction_function = tf.argmax(nn_function, 1)
correct_predictions = tf.equal(prediction_function, tf.argmax(y_, 1))
accuracy_function = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# =================
# starting the tensorflow session
# =================
session = tf.Session()
session.run(init_function)

# =================
# training phase
# =================
for epoch in range(training_epochs):
    # =================
    # optimization for each epoch
    # =================
    # reduces the loss (cost)
    session.run(training_function, feed_dict={x: train_x, y_: train_y})
    # gets the current cost with the current parameters
    current_cost = session.run(cost_function, feed_dict={x: train_x, y_: train_y})  # loss

    # =================
    # training accuracy calculation
    # =================
    # predict outputs given testing values
    predicted_y = session.run(nn_function, feed_dict={x: test_x})
    # FUNCTION: calculate the squared errors over testing samples
    mse_function = tf.reduce_mean(tf.square(predicted_y - test_y))
    # run the mse function
    current_mse = session.run(mse_function)
    # calculate the accuracy with the current nn parameters
    current_accuracy = (session.run(accuracy_function, feed_dict={x: train_x, y_: train_y}))

    # =================
    # append values
    # =================
    cost_history.append(current_cost)
    mse_history.append(current_mse)
    accuracy_history.append(current_accuracy)

    if epoch % 100 == 0:
        print("epoch: ", epoch, " - Cost: ", current_cost, "- MSE", current_mse, " - Accuracy: ", current_accuracy)

# =================
# saving the model
# =================
# create a saver object in order to save the trained model after the training
saver = tf.train.Saver()
save_path = saver.save(session, model_path)
print("Model saved in file: %s" % save_path)

# =================
# visualizing the training phase
# =================
# plot mse, accuracy and cost graphs
plt.plot(mse_history, "r")
plt.show()
plt.plot(accuracy_history)
plt.show()
plt.plot(cost_history[1:])
plt.show()

# =================
# visualizing the trained model final tests
# =================

print("Test accuracy: ", (session.run(accuracy_function, feed_dict={x: test_x, y_: test_y})))
# print the final mean square error
predicted_y = session.run(nn_function, feed_dict={x: test_x})
mse_function = tf.reduce_mean(tf.square(predicted_y - test_y))
print("MSE: %.4f" % session.run(mse_function))

session.close()

# =================
# restore the model
# =================
session_2 = tf.Session()
session_2.run(init_function)
saver.restore(session_2, model_path)


for i in range(93, 101):
    prediction_run = session_2.run(prediction_function, feed_dict={x: X[i].reshape(1, 60)})
    accuracy_run = session_2.run(accuracy_function, feed_dict={x: X[i].reshape(1, 60), y_: Y[i].reshape(1, 2)})
    print("Original class: ", Y[i], " Predicted values: ", prediction_run, " Accuracy: ", accuracy_run)
