"""
Asynchronous Advantage Actor Critic (A3C)
with discrete a space.
"""
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from A3C_NN_dis import NN_PI_s
from A3C_NN_dis import NN_V_s

class A3C_global:
    def __init__(self,
                 seed,
                 name_model,
                 scope,
                 sess,
                 n_features,
                 n_actions,
                 lr_C,
                 lr_A,
                 reward_decay):

        tf.set_random_seed(seed)

        self.name_model   = name_model
        self.sess         = sess
        self.n_features   = n_features
        self.n_actions    = n_actions
        self.reward_decay = reward_decay

        with tf.variable_scope(scope):

            self.s      = tf.placeholder(tf.float32, [None, n_features])
            self.v      = NN_V_s(self.s, 'critic')
            self.a_prob = NN_PI_s(self.s, n_actions, 'actor')

        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')

        self.OptC = tf.train.AdamOptimizer(lr_C, name='OptC')
        self.OptA = tf.train.AdamOptimizer(-lr_A, name='OptA')

    def load_model(self, saver):
        saver.restore(self.sess, "../model/"+self.name_model+".ckpt")
        #print("Model loaded from: ", "../model/" + self.name_model + ".ckpt")

    def save_model(self, saver):
        save_path = saver.save(self.sess, "../model/"+self.name_model+".ckpt")
        #print("Model saved with path: %s" % save_path)

class A3C:
    def __init__(self, scope, gAC):

        self.sess         = gAC.sess
        self.n_features   = gAC.n_features
        self.n_actions    = gAC.n_actions
        self.reward_decay = gAC.reward_decay
        self.name_model   = gAC.name_model
        self.plot_serie   = "../figures/A3C/"

        self.memoryAhead = MemoryAhead_all_n_step(self.reward_decay)

        with tf.variable_scope(scope):

            self.s        = tf.placeholder(tf.float32, [None, self.n_features])
            self.a        = tf.placeholder(tf.int32,   [None, 1])
            self.v_target = tf.placeholder(tf.float32, [None, 1])

            self.v       = NN_V_s(self.s, 'critic') #[None, 1]
            self.a_prob  = NN_PI_s(self.s, self.n_actions, 'actor') #[None, n_actions]

        # tf.squeeze removes dimensions of size 1 from the shape of a tensor.
        # aqui un vector de dimension n_samples de vectores de dimension 1 [None, 1], pasa a ser
        # un vector de n_samples [None, ]
        self.td_error_c = tf.squeeze(tf.subtract(self.v_target, self.v)) #[None, ]
        self.loss_c = (tf.reduce_mean(tf.square(self.td_error_c))) #[1, ]

        self.tmp1 = tf.one_hot(tf.squeeze(self.a), self.n_actions) #[None, n_actions]
        self.tmp2 = tf.log(self.a_prob+1e-10) * self.tmp1 #[None, n_actions]
        self.log_prob = tf.reduce_sum(self.tmp2, axis=1) #[None, ]
        self.loss_a = tf.reduce_mean(self.log_prob * tf.stop_gradient(self.td_error_c))

        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        self.c_grads = tf.gradients(self.loss_c, self.c_params)
        self.a_grads = tf.gradients(self.loss_a, self.a_params)

        self.update_c_op = gAC.OptC.apply_gradients(zip(self.c_grads, gAC.c_params))
        self.update_a_op = gAC.OptA.apply_gradients(zip(self.a_grads, gAC.a_params))

        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, gAC.c_params)]
        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, gAC.a_params)]

        self.error_history = []

    def choose_action(self, s):
        acts_prob = self.sess.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})[0]

        for action in range(self.n_actions):
            if np.isnan(acts_prob[action]):
                print("NaN!!! decrease the learning rate.")
                return np.nan

        return np.random.choice(range(self.n_actions), p=acts_prob)

    def learn(self, v_s_):
        b_s, b_a, b_v = self.memoryAhead.process(v_s_)

        feed_dict = {
            self.s:     b_s,
            self.a:     b_a,
            self.v_target:  b_v,
        }

        #td_error_c, v, loss_c = self.sess.run([self.td_error_c, self.v, self.loss_c], feed_dict)
        #tmp1, tmp2, a_prob = self.sess.run([self.tmp1, self.tmp2, self.a_prob], feed_dict)
        #log_prob, loss_a = self.sess.run([self.log_prob, self.loss_a], feed_dict)

        error = self.sess.run(self.loss_c, feed_dict)

        self.sess.run([self.update_c_op, self.update_a_op], feed_dict)
        self.sess.run([self.pull_c_params_op, self.pull_a_params_op])

        return np.sqrt(error)

    def v_s(self, s):
        return self.sess.run(self.v, {self.s: s[np.newaxis, :]})[0, 0]

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
            plt.figure("error").savefig(self.plot_serie+"_error_history.png")

        plt.close("all")

    def save_error_history(self):
        if len(self.error_history) == 0:
            return
        f = open("../model/A3C_error_history.gz", 'a+')
        for index in range(100):
            self.error_history.append(0.0)
        np.savetxt(f, self.error_history, fmt='%1.3f')
        f.close()
        self.error_history = []

    def load_error_history(self):
        f = open("../model/A3C_error_history.gz", 'r')
        self.error_history = np.loadtxt(f, dtype=float)
        f.close()

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

        acts_prob = self.sess.run(self.a_prob, {self.s: states})

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

    def plot_V_a_1D_DataCenter(self, episode, do_not_save):

        state_res          = [0,1,2,3,4,5,6,7,8,9,10]
        state_req          = [0,1,2,3]
        n_state_res        = len(state_res)
        n_state_req        = len(state_req)
        states             = np.zeros((n_state_res, 2))
        actions_numerical  = np.zeros((n_state_req, n_state_res, 1))
        v                  = np.zeros((n_state_req, n_state_res, 1))
        num_actions        = self.n_actions

        for index_state_req in range(n_state_req):

            for index_state_res in range(n_state_res):
                states[index_state_res, :] = state_req[index_state_req], state_res[index_state_res]

            v[index_state_req, :] = self.sess.run(self.v, {self.s: states})
            acts_prob = self.sess.run(self.a_prob, {self.s: states})

            for index in range(n_state_res):
                #actions_numerical[index_state_req, index, 0] = np.random.choice(range(num_actions), p=acts_prob[index].ravel())
                actions_numerical[index_state_req, index, 0] = np.argmax(acts_prob[index])

        if do_not_save:
            plt.figure("V_1D")
            plt.suptitle("A3C V function, episode: " + str(episode))
        else:
            plt.figure("V_1D", figsize=(19.2, 10.8), dpi=100)
            plt.suptitle("A3C V function, episode: " + str(episode))

        colors = ["blue", "green", "yellow", "red"]
        for index_state_req in range(n_state_req):
            plt.plot(state_res, v[index_state_req], c=colors[index_state_req])
        plt.xlabel('free servers')
        #plt.grid()

        if do_not_save:
            plt.figure("a_1D")
            plt.suptitle("A3C a function, episode: " + str(episode))
        else:
            plt.figure("a_1D", figsize=(19.2, 10.8), dpi=100)
            plt.suptitle("A3C a function, episode: " + str(episode))

        for index_state_req in range(n_state_req):
            plt.plot(state_res, actions_numerical[index_state_req], c=colors[index_state_req])
        plt.xlabel('free servers')
        #plt.grid()

        if do_not_save:
            plt.show()
        else:
            plt.figure("V_1D").savefig(self.plot_serie + "_V_" + str(episode) + ".png")
            plt.figure("a_1D").savefig(self.plot_serie + "_a_" + str(episode) + ".png")

    def plot_graph(self, data, title, do_not_save):
        if do_not_save:
            plt.figure(title)
        else:
            plt.figure(title, figsize=(19.2, 10.8), dpi=100)
        plt.plot(range(len(data)), data)
        plt.ylabel(title)
        plt.xlabel('steps')
        plt.grid()
        # fig, ax = plt.subplots()
        # ax.set_yticks([0,20])
        # plt.yticks(range(0,21))
        if do_not_save:
            plt.show()
        else:
            plt.figure(title).savefig(self.plot_serie + title + self.name_model + ".png")

        #plt.close("all")


#####################  Memory-ahead  ####################

class MemoryAhead_all_n_step:

    def __init__(self, gamma):
        self.b_s = []
        self.b_a = []
        self.b_r = []
        self.b_v = []
        self.gamma = gamma

    def store_transition(self, s, a, r):
        self.b_s.append(s)
        self.b_a.append(a)
        self.b_r.append(r)

    def process(self, v_s_):
        v_s = v_s_
        self.b_r.reverse()
        for r in self.b_r:
            v_s = r + self.gamma * v_s
            self.b_v.append(v_s)
        self.b_v.reverse()

        b_s = np.vstack(self.b_s)
        b_a = np.vstack(self.b_a)
        b_v = np.vstack(self.b_v)

        self.b_s = []
        self.b_a = []
        self.b_r = []
        self.b_v = []

        return b_s, b_a, b_v

class MemoryAhead_last_n_step:

    def __init__(self, gamma):
        self.b_s = []
        self.b_a = []
        self.b_r = []
        self.b_v = []
        self.gamma = gamma

    def store_transition(self, s, a, r):
        self.b_s.append(s)
        self.b_a.append(a)
        self.b_r.append(r)

    def process(self, v_s_):
        v_s = v_s_
        self.b_r.reverse()
        for r in self.b_r:
            v_s = r + self.gamma * v_s
            self.b_v.append(v_s)
        self.b_v.reverse()

        b_s = np.vstack([self.b_s[0]])
        b_a = np.vstack([self.b_a[0]])
        b_v = np.vstack([self.b_v[0]])

        self.b_s = []
        self.b_a = []
        self.b_r = []
        self.b_v = []

        return b_s, b_a, b_v

############################################################
