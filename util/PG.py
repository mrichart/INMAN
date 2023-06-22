"""
Policy Gradient.
"""

import os
import sys
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from PG_NN import NN_PI_s

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(
            self,
            n_features,
            n_actions,
            learning_rate=0.01,
            reward_decay=0.99,
            to_load_model=False,
            name_model=None
    ):

        tf.disable_eager_execution()

        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay

        self.memory = Memory()

        self.to_load_model = to_load_model
        self.name_model = name_model

        self.s = tf.placeholder(tf.float32, [None, n_features])
        self.a = tf.placeholder(tf.int32,   [None, ]) # array de enteros, así one_hot funciona bien.
        self.q = tf.placeholder(tf.float32, [None, ]) # array de floats

        self.acts_prob = NN_PI_s(self.s, self.n_actions, scope='net_params') #[None, n_actions]

        # one_hot es un array de tamaño n_steps, cada elemento es un sub-array de tamaño n_actions,
        # donde se pone a 1 la posición a de n_actions,

        # el producto: tf.log(self.acts_prob+1e-10) * tf.one_hot(self.a, self.n_actions)
        # es un array de tamaño n_steps, cada elemento es un sub-array de tamaño n_actions,
        # resultado del producto elemento a elemento de los sub-arrays: act_prob * one_hot,
        # el sub-array resultante son ceros menos para la acción tomada que sera su probabilidad.

        # reduce_sum con axis = 1, es el array de tamaño n_steps, donde cada elemento es un float,
        # que resulta de la suma de los elementos del sub-array de tamaño n_actions.

        # reduce_mean es la suma de las contribuciones de todos los steps dividida por n_steps.

        # to maximize the value of s_0 is to use the gradients of: (log(prob(a|s)) * Q(s,a)) for the
        # sequence of all states until a terminal state is reached or the state where we cut:

        self.tmp1 = tf.log(self.acts_prob+1e-10) #[None, n_actions]
        self.tmp2 = tf.one_hot(self.a, self.n_actions) #[None, n_actions]
        self.log_prob = tf.reduce_sum(self.tmp1 * self.tmp2, axis=1) #[None, ]
        self.loss     = tf.reduce_mean(self.log_prob * self.q) #[1, ]

        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net_params')
        self.OptA     = tf.train.AdamOptimizer(-self.lr)

        self.a_grads  = tf.gradients(self.loss, self.a_params)
        self.train_op = self.OptA.apply_gradients(zip(self.a_grads, self.a_params))

        self.sess  = tf.Session()
        self.saver = tf.train.Saver()

        if to_load_model:
            self.saver.restore(self.sess, "./model/"+name_model+".ckpt")
            print("Model loaded from: ", "./model/" + self.name_model + ".ckpt")
        else:
            self.sess.run(tf.global_variables_initializer())

        self.Q_s0_history = []

    def choose_action(self, s):
        act_prob = self.sess.run(self.acts_prob, feed_dict={self.s: s[np.newaxis, :]})[0]
        for action in range(self.n_actions):
            if np.isnan(act_prob[action]):
                print("NaN!!! decrease the learning rate.")
                return np.nan
        #return np.random.choice(range(self.n_actions), p=act_prob)
        return np.argmax(act_prob)

    def learn(self):

        epi_q = self.memory.q(self.gamma)

        feed_dict = {
            self.s: self.memory.epi_s,
            self.a: self.memory.epi_a,
            self.q: epi_q,
        }

        # train on a whole episode
        #tmp1, tmp2, log_prob = self.sess.run([self.tmp1, self.tmp2, self.log_prob], feed_dict)
        self.sess.run([self.train_op], feed_dict)

        self.Q_s0_history.append(epi_q[0])
        self.memory.empty() # empty episode data

    def save_model(self):
        save_path = self.saver.save(self.sess, "./model/"+self.name_model+".ckpt")
        print("Model saved in path: %s" % save_path)

    def plot_graph(self, data, ylabel, xlabel):
        plt.plot(range(len(data)), data)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.grid()
        plt.show()

    def plot_a_2D_DataCenter(self, episode, do_not_save):

        state_res = [1,2,3,4,5,6,7,8,9,10]
        state_req = [0,1,2,3]
        n_state_res = len(state_res)
        n_state_req = len(state_req)
        states = np.zeros((n_state_res * n_state_req, 2))
        actions_numerical = np.zeros((n_state_res * n_state_req, 1))
        num_actions = self.n_actions

        index = 0
        for index_state_req in range(n_state_req):
            for index_state_res in range(n_state_res):
                states[index, :] = state_req[index_state_req], state_res[index_state_res]
                index += 1

        acts_prob = self.sess.run(self.acts_prob, {self.s: states})

        for index in range(n_state_res * n_state_req):
            #a = np.random.choice(range(num_actions), p=acts_prob[index].ravel())
            #actions_numerical[index, :] = a
            actions_numerical[index, :] = np.argmax(acts_prob[index])

        fig2 = plt.figure("a_2D")
        x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y2 = [0, 1, 2, 3]
        X2, Y2 = np.meshgrid(x2, y2)
        plt.scatter(np.ravel(X2), np.ravel(Y2), s=128, c=np.ravel(actions_numerical))
        plt.title("A3C a function, episode: "+str(episode))
        plt.xlabel("free servers")
        plt.ylabel("inverted priority")
        #plt.grid()

        if do_not_save:
            plt.show()
        else:
            plt.figure("a_2D").savefig(self.plot_serie + "_2D_a_" + str(episode) + ".png")

        #plt.close("all")

#####################  Memory  ####################

class Memory(object):
    def __init__(self):
        self.epi_s = []
        self.epi_a = []
        self.epi_r = []

    def store_transition(self, s, a, r):
        self.epi_s.append(s)
        self.epi_a.append(a)
        self.epi_r.append(r)

    def q(self, gamma):
        q = np.zeros(len(self.epi_r))
        q[-1] = self.epi_r[-1] * (1/(1-gamma)) # this is because we have to add v(s_last) when we cut
        for t in reversed(range(0, len(self.epi_r)-1)):
            q[t] = self.epi_r[t] + (gamma * q[t+1])
        #q -= np.mean(q)
        #q /= np.std(q)
        return q

    def empty(self):
        self.epi_s = []
        self.epi_a = []
        self.epi_r = []

############################################################
