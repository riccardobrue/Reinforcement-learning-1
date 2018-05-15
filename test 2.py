import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create two clusters of red points centered at (0, 0) and (1, 1), respectively.
red_points = np.concatenate((
    0.2*np.random.randn(25, 2) + np.array([[0, 0]]*25),
    0.2*np.random.randn(25, 2) + np.array([[1, 1]]*25)
))

# Create two clusters of blue points centered at (0, 1) and (1, 0), respectively.
blue_points = np.concatenate((
    0.2*np.random.randn(25, 2) + np.array([[0, 1]]*25),
    0.2*np.random.randn(25, 2) + np.array([[1, 0]]*25)
))

# Create training input placeholder
X = tf.placeholder(dtype=tf.float64)

# Create placeholder for the training classes
c = tf.placeholder(dtype=tf.float64)

# Build a hidden layer
W_hidden = tf.Variable(np.random.randn(2, 2))
b_hidden = tf.Variable(np.random.randn(2))
p_hidden = tf.sigmoid( tf.add(tf.matmul(X, W_hidden), b_hidden) )

# Build the output layer
W_output = tf.Variable(np.random.randn(2, 2))
b_output = tf.Variable(np.random.randn(2))
p_output = tf.nn.softmax( tf.add(tf.matmul(p_hidden, W_output), b_output) )

# Build cross-entropy loss
J = tf.negative(tf.reduce_sum(tf.reduce_sum(tf.multiply(c, tf.log(p_output)), axis=1)))

# Build minimization op
minimization_op = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(J)

# Build placeholder inputs
feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    c:
        [[1, 0]] * len(blue_points)
        + [[0, 1]] * len(red_points)

}

# Create session
session = tf.Session()

# Initialize variables
session.run(tf.global_variables_initializer())

# Perform 1000 gradient descent steps
for step in range(1000):
    J_value = session.run(J, feed_dict)
    if step % 100 == 0:
        print("Step:", step, " Loss:", J_value)
    session.run(minimization_op, feed_dict)

# Print final result
W_hidden_value = session.run(W_hidden)
print("Hidden layer weight matrix:\n", W_hidden_value)
b_hidden_value = session.run(b_hidden)
print("Hidden layer bias:\n", b_hidden_value)
W_output_value = session.run(W_output)
print("Output layer weight matrix:\n", W_output_value)
b_output_value = session.run(b_output)
print("Output layer bias:\n", b_output_value)

# Visualize classification boundary
xs = np.linspace(-2, 2)
ys = np.linspace(-2, 2)
pred_classes = []
for x in xs:
    for y in ys:
        pred_class = session.run(p_output,
                              feed_dict={X: [[x, y]]})[0]
        pred_classes.append((x, y, pred_class.argmax()))
xs_p, ys_p = [], []
xs_n, ys_n = [], []
for x, y, c in pred_classes:
    if c == 0:
        xs_n.append(x)
        ys_n.append(y)
    else:
        xs_p.append(x)
        ys_p.append(y)
plt.plot(xs_p, ys_p, 'ro', xs_n, ys_n, 'bo')