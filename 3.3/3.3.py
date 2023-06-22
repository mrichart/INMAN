import numpy as np
import matplotlib.pyplot as plt

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

number_of_points = 50
x_points = []
y_points = []
a = 0.22
b = 0.78

for i in range(number_of_points):
    x = np.random.normal(0.0, 1.0)
    y = a * x + b + np.random.normal(0.0, 0.1)
    x_points.append([x])
    y_points.append([y])

plt.plot(x_points, y_points, 'o')
plt.show()

x        = tf.placeholder(tf.float32, shape=[None, 1])
y_target = tf.placeholder(tf.float32, shape=[None, 1])

w0 = tf.Variable(tf.zeros(())) # shape = ()
w1 = tf.Variable(tf.random_uniform(shape=[], minval=-1.0, maxval=1.0))

y = w1*x + w0
error = y_target - y

cost_function = tf.reduce_mean(tf.square(error))
optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(cost_function)

serie_of_y_points = []
step = 0

with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    while True:

        w0_old = session.run(w0)
        w1_old = session.run(w1)
        session.run(train, {x: x_points, y_target: y_points})
        w0_new = session.run(w0)
        w1_new = session.run(w1)
        print("w1: ", w1_new, ", w0: ", w0_new)

        step +=1
        if (step % 10) == 0:
            serie_of_y_points.append(session.run(y, {x: x_points}))
            plt.plot(x_points, y_points, 'o')
            for index in range(len(serie_of_y_points)):
                plt.plot(x_points, serie_of_y_points[index])
            plt.show()

        if abs(w1_new-w1_old)+abs(w0_new-w0_old) < 0.001:
            break

    plt.plot(x_points, y_points, 'o')
    plt.plot(x_points, session.run(y, {x: x_points}))
    plt.show()
