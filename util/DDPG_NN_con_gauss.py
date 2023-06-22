import tensorflow as tf

def NN_PI_s(s, n_actions, action_bound, scope, trainable):

    with tf.variable_scope(scope):

        l1 = tf.layers.dense(
            inputs=s,
            units=100,  # number of hidden units, 10 before
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .3),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1',
            trainable=trainable
        )

        mu = tf.layers.dense(
            inputs=l1,
            units=n_actions,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu',
            trainable = trainable
        )

        sigma = tf.layers.dense(
            inputs=l1,
            units=n_actions,
            activation=tf.nn.softplus,  # get value [0, same input if big one]
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='sigma',
            trainable = trainable
        )

    return mu, sigma


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