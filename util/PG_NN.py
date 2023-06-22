#import tensorflow as tf
import tensorflow.compat.v1 as tf

def NN_PI_s(s, n_actions, scope):

    with tf.variable_scope(scope):

        layer1 = tf.layers.dense(
            inputs=s,
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer1'
        )

        acts_prob = tf.layers.dense(
            inputs=layer1,
            units=n_actions,
            activation=tf.nn.softmax,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='acts_prob'
        )

    return acts_prob