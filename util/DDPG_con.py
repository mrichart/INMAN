'''
Deep Deterministic Policy Gradient (DDPG).
'''
import sys
import tensorflow as tf
import numpy as np
import gym
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from DDPG_NN_con import NN_PI_s
from DDPG_NN_con import NN_Q_s_a

np.random.seed(1)
tf.set_random_seed(1)

class DeepDeterministicPolicyGradient:

    def __init__(self,
                 n_features,
                 n_actions,
                 action_bound,
                 replace,
                 lr_c=0.01,
                 lr_a=0.001,
                 reward_decay=0.95,
                 memory_size=1024,
                 batch_size=128,
                 to_load_model=False,
                 name_model=None,
                 plot_serie="./figures/DDPG/",
        ):

        self.plot_serie = plot_serie

        self.n_features = n_features
        self.n_actions = n_actions
        self.action_bound = action_bound
        self.lr_c = lr_c
        self.lr_a = lr_a
        self.gamma = reward_decay

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = Memory(memory_size, n_features * 2 + n_actions + 1)

        self.replace = replace

        self.to_load_model = to_load_model
        self.name_model = name_model

        self.s  = tf.placeholder(tf.float32, shape=[None, n_features])
        self.s_ = tf.placeholder(tf.float32, shape=[None, n_features])
        self.r  = tf.placeholder(tf.float32, shape=[None, 1])

        with tf.variable_scope('Actor_con'):
            self.a  = NN_PI_s(self.s,  n_actions, action_bound,  'eval_net', trainable=True) #[None, 1]
            self.a_ = NN_PI_s(self.s_, n_actions, action_bound, 'target_net', trainable=False) #[None, 1]

        with tf.variable_scope('Critic_con'):
            self.q  = NN_Q_s_a(self.s,  self.a,  n_features, n_actions, 'eval_net',   trainable=True) #[None, 1]
            self.q_ = NN_Q_s_a(self.s_, self.a_, n_features, n_actions, 'target_net', trainable=False) #[None, 1]

        self.a_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor_con/eval_net')
        self.a_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor_con/target_net')
        self.c_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic_con/eval_net')
        self.c_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic_con/target_net')

        self.q_target = self.r + self.gamma * self.q_ #[None, 1]
        self.loss_c = tf.reduce_mean(tf.squared_difference(self.q_target, self.q)) #[1, ]
        self.train_op_c = tf.train.AdamOptimizer(self.lr_c).minimize(self.loss_c)

        self.q_grads = tf.gradients(self.q, self.a)  # tensor of gradients of each sample [None, a_vec]
        # tf.gradients calculates dys/dxs multiplied by grad_ys, so this is dq/da * da/dparams, hence,
        # grad_ys and ys must have the same dimensions, yes, dq/da has as many dimensions as a.
        self.a_grads = tf.gradients(ys=self.a, xs=self.a_e_params, grad_ys=self.q_grads)
        opt = tf.train.AdamOptimizer(-self.lr_a) # (- learning rate) for ascent policy
        self.train_op_a = opt.apply_gradients(zip(self.a_grads, self.a_e_params))

        if self.replace['name'] == 'hard':
            self.replace_counter = 0
            self.hard_replace_C = [tf.assign(t, e) for t, e in zip(self.c_t_params, self.c_e_params)]
            self.hard_replace_A = [tf.assign(t, e) for t, e in zip(self.a_t_params, self.a_e_params)]
        else:
            self.soft_replace_C = [tf.assign(t, (1 - self.replace['tau']) * t + self.replace['tau'] * e)
                                   for t, e in zip(self.c_t_params, self.c_e_params)]
            self.soft_replace_A = [tf.assign(t, (1 - self.replace['tau']) * t + self.replace['tau'] * e)
                                   for t, e in zip(self.a_t_params, self.a_e_params)]
            
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        if to_load_model:
            self.load_model()
        else:
            self.sess.run(tf.global_variables_initializer())

        self.error_history = []

    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={self.s: s[np.newaxis, :]})[0]

    def learn(self):   # batch update

        samples = self.memory.sample(self.batch_size)
        n_features = self.n_features
        n_actions  = self.n_actions
        s  = samples[:, :n_features]
        a  = samples[:,  n_features : n_features + n_actions] #[None, 1]
        r  = samples[:, -n_features - 1 : -n_features]        #[None, 1]
        s_ = samples[:, -n_features : ]

        # it is necessary to record a, as it was the a for the policy of that time
        # the policy is different now, when used for training (off-policy), besides
        # we don't want to learn pi(s|theta), we want to learn q(s,a|theta), s and a must be constants:
        _, error = self.sess.run([self.train_op_c, self.loss_c], feed_dict={self.s: s, self.a: a, self.r: r, self.s_: s_})

        #self.error_history.append(error)

        if self.replace['name'] == 'soft':
            self.sess.run(self.soft_replace_C)
        else:
            if self.replace_counter % self.replace['rep_iter_c'] == 0:
                self.sess.run(self.hard_replace_C)

        # the input data is only s, not a, because the a has to be pi(s), not the a used at that time,
        # this is because we want dq/dtheta = dq/da * da/dtheta, the value of a must be the same
        # in both partial derivatives:
        self.sess.run(self.train_op_a, feed_dict={self.s: s})

        if self.replace['name'] == 'soft':
            self.sess.run(self.soft_replace_A)
        else:
            if self.replace_counter % self.replace['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace_A)
            self.replace_counter += 1

        return error

    def save_model(self):
        save_path = self.saver.save(self.sess, "./model/"+self.name_model+".ckpt")

    def load_model(self):
        self.saver.restore(self.sess, "./model/"+self.name_model+".ckpt")
        print("Model loaded from: ", "./model/"+self.name_model+".ckpt")

    def plot_error_history(self, do_not_save):

        if do_not_save:
            plt.figure("error")
        else:
            plt.figure("error", figsize=(19.2, 10.8), dpi=100)
        plt.plot(range(len(self.error_history)), self.error_history)
        plt.ylabel('error')
        plt.xlabel('training steps')
        plt.grid()
        if do_not_save:
            plt.show()
        else:
            plt.figure("error").savefig(self.plot_serie + "_error_history.png")

        plt.close("all")

    def save_error_history(self):
        if len(self.error_history) == 0:
            return
        f = open("./model/"+self.name_model+"_error_history.gz", 'a+')
        for index in range(100):
            self.error_history.append(0.0)
        np.savetxt(f, self.error_history, fmt='%1.3f')
        f.close()
        self.error_history = []

    def load_error_history(self):
        f = open("./model/"+self.name_model + "_error_history.gz", 'r')
        self.error_history = np.loadtxt(f, dtype=float)
        f.close()

    def plot_data(self, data, title, do_not_save, labels=[]):
        if do_not_save:
            plt.figure(title)
        else:
            plt.figure(title, figsize=(19.2, 10.8), dpi=100)
        if data.ndim==1:
            plt.plot(range(len(data)), data, 'o', markersize=2, label=title)
        else:
            for graph in range(data.shape[1]):
                plt.plot(range(len(data)), data[:, graph], 'o', ms=2, label=labels[graph])
        plt.ylabel(title)
        plt.xlabel('steps')
        plt.legend()
        plt.grid()
        if do_not_save:
            plt.show()
        else:
            plt.figure(title).savefig(self.plot_serie + title + self.name_model + ".png")

        plt.close("all")

    def print_success(self, success_ratio, steps_in_episode):
        plt.figure(1)
        axes = plt.gca()
        axes.set_ylim([0.00, 1.00])
        plt.plot(range(len(success_ratio)), success_ratio)
        plt.xlabel('episode')
        plt.ylabel('success_ratio')
        plt.figure(2)
        axes = plt.gca()
        axes.set_ylim([0, 10000])
        plt.plot(range(len(steps_in_episode)), steps_in_episode)
        plt.xlabel('episode')
        plt.ylabel('steps_in_episode')
        plt.show()

#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, batch_size):
        if self.pointer > self.capacity:
            samples_index = np.random.choice(self.capacity, size=batch_size)
        else:
            samples_index = np.random.choice(self.pointer, size=batch_size)

        return self.data[samples_index, :]

############################################################


