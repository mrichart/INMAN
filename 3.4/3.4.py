#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math, random
import matplotlib.pyplot as plt

np.random.seed(1000)

def function_to_learn(x):
    return np.cos(x) + 0.1*np.random.randn(*x.shape)

z = np.array([[1,2,3],[1,2,3]])
print(z.shape)
print(*z.shape)

NUM_points = 1000

all_x = np.float32(np.random.uniform(-2*math.pi, 2*math.pi, (NUM_points,1)))

train_size = int(NUM_points*0.7)
x_training = all_x[:train_size]
y_training = function_to_learn(x_training)

#the last 300 are in the validation set
x_validation = all_x[train_size:]
y_validation = function_to_learn(x_validation)

plt.scatter(x_training, y_training, c='green', label='train')
plt.scatter(x_validation, y_validation, c='red', label='validation')
plt.legend()
plt.show()

#############################################################

X = tf.placeholder(tf.float32, [None, 1], name="X")
print("X shape: ", X.shape)
Y = tf.placeholder(tf.float32, [None, 1], name="Y")
print("Y shape: ", Y.shape)

layer_1_neurons = 10

#hidden layer
w_h = tf.Variable(tf.random_uniform([1, layer_1_neurons], -1.0, 1.0))
print("w_h shape: ", w_h.shape) # shape (1,10)
b_h = tf.Variable(tf.zeros([1, layer_1_neurons]))
print("b_h shape: ", b_h.shape) # shape (1,10)
h   = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)
print("h shape: ", h.shape) # shape (?,10)

#output layer
w_o = tf.Variable(tf.random_uniform([layer_1_neurons, 1], -1.0, 1.0))
print("w_o shape: ", w_o.shape) # shape (10,1)
b_o = tf.Variable(tf.zeros([1, 1]))
print("b_o shape: ", b_o.shape) # shape (1,1)
output = tf.matmul(h, w_o) + b_o
print("output shape: ", output.shape) # shape (?,1)

error  = Y - output
#MSE = tf.nn.l2_loss(error)
MSE = tf.reduce_mean(tf.square(error))

#minimize the MSE
train_op = tf.train.AdamOptimizer().minimize(MSE)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 100
steps = 10000

errors = []
for i in range(steps):

    indexes = np.random.choice(len(x_training), size=batch_size)
    sess.run(train_op, {X: x_training[indexes],
                        Y: y_training[indexes]})

    mse = sess.run(MSE, {X:x_validation, Y: y_validation})
    errors.append(mse)
    if i%1000 == 0: print ("the_step %d, MSE = %g" % (i, mse))

plt.plot(errors, label='MLP Function Approximation')
plt.xlabel('steps')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.scatter(x_training, y_training, c='green', label='train')
plt.scatter(x_validation, y_validation, c='red', label='validation')
plt.scatter(all_x, sess.run(output, {X: all_x}), label='aproximation')
plt.legend()
plt.show()

