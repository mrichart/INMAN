#import tensorflow as tf
import tensorflow.compat.v1 as tf

def NN_PI_s(s, n_actions, scope):

    with tf.variable_scope(scope):

        l1 = tf.layers.dense(inputs=s,
                             units=10,
                             activation=tf.nn.tanh,
                             kernel_initializer=tf.random_normal_initializer(0., 0.3),
                             bias_initializer=tf.constant_initializer(0.1),
                             name=scope + "l1")

        l2 = tf.layers.dense(inputs=l1,
                             units=10,
                             activation=tf.nn.tanh,
                             kernel_initializer=tf.random_normal_initializer(0., 0.3),
                             bias_initializer=tf.constant_initializer(0.1),
                             name=scope + "l2")

        acts_prob = tf.layers.dense(
            inputs=l2,
            units=n_actions,
            activation=tf.nn.softmax,
            kernel_initializer=tf.random_normal_initializer(0., .3),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name=scope + 'acts_prob'
        )

    return acts_prob


def NN_V_s(s, scope):

    with tf.variable_scope(scope):

        l1 = tf.layers.dense(inputs=s,
                             units=10,
                             activation=tf.nn.tanh,
                             kernel_initializer=tf.random_normal_initializer(0., 0.3),
                             bias_initializer=tf.constant_initializer(0.1),
                             name=scope + "l1")

        l2 = tf.layers.dense(inputs=l1,
                             units=10,
                             activation=tf.nn.tanh,
                             kernel_initializer=tf.random_normal_initializer(0., 0.3),
                             bias_initializer=tf.constant_initializer(0.1),
                             name=scope + "l2")

        V = tf.layers.dense(
            inputs=l2,
            units=1,
            kernel_initializer=tf.random_normal_initializer(0., .3),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name=scope + 'V'
        )

    return V