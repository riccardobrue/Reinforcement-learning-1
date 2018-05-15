"""
https://www.youtube.com/watch?v=yX8KuPZCAMo
"""

import tensorflow as tf

node_1 = tf.constant(5.0, tf.float32)  # specifying the type
node_2 = tf.constant(6.0)

out_mul = node_1 * node_2

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# print(node_1, node_2)

adder_node = a + b

# Variables --> creation of a model
# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Inputs and outputs
x = tf.placeholder(tf.float32)  # inputs
y = tf.placeholder(tf.float32)  # outputs (already known)

linear_model = W * x + b  # model function (linear model function)

# loss (cost/error)
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)  # loss function

# Optimize
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()  # initialize all the Variables

# session = tf.Session()
# print(session.run([node_1, node_2]))
# session.close() #needs to close explicitly

with tf.Session() as session:  # auto-closing the session
    # output = session.run([node_1, node_2])
    # print(output)
    # print(session.run(out_mul))

    # print(session.run(adder_node, {a: [1, 3], b: [2, 4]}))  # using placeholders
    session.run(init)
    # print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
    for i in range(1000):
        # session.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
        session.run(train, feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3]})  # more optimized
    print(session.run([W, b]))
