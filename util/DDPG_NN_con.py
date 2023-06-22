import tensorflow as tf

def NN_PI_s(s, n_actions, action_bound, scope, trainable):

    with tf.variable_scope(scope):

        l1 = tf.layers.dense(
            inputs=s,
            units=100,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .3),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1',
            trainable=trainable
        )

        action_normalized = tf.layers.dense(
            inputs=l1,
            units=n_actions,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .3),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='action_normalized',
            trainable=trainable
        )

        # using activation=tf.nn.relu6:
        # output = tf.add(action_normalized, tf.constant(-3.0))
        # scaled_a = tf.multiply(output, action_bound/3, name='scaled_a')

        scaled_a = tf.multiply(action_normalized, action_bound, name='scaled_a')

    return scaled_a


def NN_Q_s_a(s, a, s_dim, a_dim, scope, trainable):

    with tf.variable_scope(scope):
        s_a = tf.concat([s, a], 1)

        l1 = tf.layers.dense(
            inputs=s_a,
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .3),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1',
            trainable=trainable
        )
        q = tf.layers.dense(
            inputs=l1,
            units=1,
            kernel_initializer=tf.random_normal_initializer(0., .3),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            trainable=trainable)
    return q