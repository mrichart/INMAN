
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def NN_Q_s(s, n_actions, scope):

    with tf.variable_scope(scope):
        l1 = tf.layers.dense(inputs=s,
                units=10,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., 0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name=scope+"l1")

        l2 = tf.layers.dense(inputs=l1,
                             units=10,
                             activation=tf.nn.tanh,
                             kernel_initializer=tf.random_normal_initializer(0., 0.3),
                             bias_initializer=tf.constant_initializer(0.1),
                             name=scope+"l2")

        q_s_a = tf.layers.dense(inputs=l2,
                units=n_actions,
                kernel_initializer=tf.random_normal_initializer(0., 0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name=scope+"q")
    return q_s_a