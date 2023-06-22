"""
Double Deep_Q_Network.
"""
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from DQN_NN import NN_Q_s

class DeepQNetwork:

    def __init__(
            self,
            n_features,
            n_actions,
            learning_rate=0.01,
            reward_decay=0.95,
            memory_size=1024,
            batch_size=128,
            replace_target_iter=300,
            to_load_model=False,
            name_model=None,
            id="0",
            plot_serie="../figures/DQN/",
    ):

        self.plot_serie = plot_serie

        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.reward_decay = reward_decay

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = Memory(memory_size, n_features * 2 + 2)

        self.replace_target_iter = replace_target_iter
        self.replace_counter = 0
        
        self.to_load_model = to_load_model
        self.name_model = name_model

        self.s          = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.s_         = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.q_target   = tf.placeholder(tf.float32, shape=[None, self.n_actions])

        self.q_s  = NN_Q_s(self.s, self.n_actions, scope='eval_net_params' + id) #(None, n_actions)
        self.q_s_ = NN_Q_s(self.s_, self.n_actions, scope='target_net_params' + id) #(None, n_actions)

        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net_params' + id)
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net_params' + id)

        self.tmp = tf.squared_difference(self.q_target, self.q_s) #(None, n_actions)
        self.error = tf.sqrt(self.n_actions * tf.reduce_mean(self.tmp)) #(1, ) (n_actions-1) are zeros in each sub-array for this mean.
        self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.error)

        self.tau = 1.0E-3
        self.replace_hard_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.replace_soft_op = [tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        if to_load_model:
            self.load_model()
        else:
            self.sess.run(tf.global_variables_initializer())

        self.error_history = []


    def choose_action(self, s):
        actions_value = self.sess.run(self.q_s, feed_dict={self.s: s[np.newaxis, :]})[0]
        return np.argmax(actions_value)

    def learn(self):
        samples  = self.memory.samples(self.batch_size)
        q_s      = self.sess.run(self.q_s, {self.s: samples[:, :self.n_features]} ) #(None, n_actions)
        q_s_     = self.sess.run(self.q_s_, {self.s_: samples[:, -self.n_features:]} ) #(None, n_actions)
        #q_sFORs_ : q_s FOR s_, that is, q_s using the states s'
        q_sFORs_ = self.sess.run(self.q_s, {self.s: samples[:, -self.n_features:]})  # (None, n_actions)
        q_target = q_s.copy() #(None, n_actions)

        samples_indices = np.arange(self.batch_size, dtype=np.int32)
        actions = samples[:, self.n_features].astype(int) #(None, )
        rewards = samples[:, self.n_features + 1] #(None, )
        #tmp1 = np.max(q_s_, axis=1) #(None, )
        tmp1 = q_s_[samples_indices, np.argmax(q_sFORs_, axis=1)]
        tmp2 = self.reward_decay * tmp1 #(None, )
        tmp3 = rewards + tmp2 #(None, )
        q_target[samples_indices, actions] = tmp3

        feed_dict = {self.s: samples[:, :self.n_features], self.q_target: q_target}
        _, error, tmp = self.sess.run([self.train_op, self.error, self.tmp], feed_dict)

        #self.replace_counter += 1
        #if self.replace_counter % self.replace_target_iter == 0:
        #    self.sess.run(self.replace_hard_op)
        self.sess.run(self.replace_soft_op)

        return error

    def save_model(self):
        save_path = self.saver.save(self.sess, "../model/"+self.name_model+".ckpt")
        #print("Model saved with path: %s" % save_path)

    def load_model(self):
        self.saver.restore(self.sess, "../model/"+self.name_model+".ckpt")
        #print("Model loaded from: ", self.dir + "/output/" + self.name_model + ".ckpt")

    def reset_model(self):
        self.sess.run(tf.global_variables_initializer())
        self.error_history = []

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
    def plot_a_2D_DataCenter(self, episode, do_not_save):

        state_res = [1,2,3,4,5,6,7,8,9,10]
        state_req = [0,1,2,3]
        n_state_res = len(state_res)
        n_state_req = len(state_req)
        states = np.zeros((n_state_res * n_state_req, 2))

        a = np.zeros((n_state_res * n_state_req, 1))
        q = np.zeros((n_state_res * n_state_req, 1))

        index = 0
        for index_state_req in range(n_state_req):
            for index_state_res in range(n_state_res):
                states[index, :] = state_req[index_state_req], state_res[index_state_res]
                index += 1

        q_s_a = self.sess.run(self.q_s, {self.s: states})

        for index in range(n_state_res * n_state_req):
            a[index, :] = np.argmax(q_s_a[index])
            q[index, :] = q_s_a[index][int(a[index])]

        fig2 = plt.figure("a_2D")
        x2 = [1,2,3,4,5,6,7,8,9,10]
        y2 = [1,2,3,4]
        X2, Y2 = np.meshgrid(x2, y2)
        plt.scatter(np.ravel(X2), np.ravel(Y2), s=128, c=np.ravel(a))
        plt.title("DQN a function, episode: "+str(episode))
        plt.xlabel("free servers")
        plt.ylabel("inverted priority")

        if do_not_save:
            plt.show()
        else:
            plt.figure("a_2D").savefig(self.plot_serie + "_2D_a_" + str(episode) + ".png")

        plt.close("all")
    def plot_q_a_1D_DataCenter(self, episode, do_not_save):

        state_res           = [0,1,2,3,4,5,6,7,8,9,10]
        state_req           = [0,1,2,3]
        n_state_res         = len(state_res)
        n_state_req         = len(state_req)
        states              = np.zeros((n_state_res, 2))

        a  = np.zeros((n_state_req, n_state_res, 1))
        q  = np.zeros((n_state_req, n_state_res, 1))

        for index_state_req in range(n_state_req):

            for index_state_res in range(n_state_res):
                states[index_state_res, :] = state_req[index_state_req], state_res[index_state_res]

            q_s_a = self.sess.run(self.q_s, {self.s: states})

            for index in range(n_state_res):
                a [index_state_req, index, 0] = np.argmax(q_s_a[index])
                q [index_state_req, index, 0] = q_s_a[index][int(a[index_state_req, index, 0])]

        if do_not_save:
            plt.figure("q_1D")
            plt.suptitle("DQN q function, episode: " + str(episode))
        else:
            plt.figure("q_1D", figsize=(19.2, 10.8), dpi=100)
            plt.suptitle("DQN q function, episode: " + str(episode))

        colors = ["blue", "green", "yellow", "red"]
        for index_state_req in range(n_state_req):
            plt.plot(state_res, q[index_state_req], c=colors[index_state_req])
        plt.xlabel('free servers')

        if do_not_save:
            plt.figure("a_1D")
            plt.suptitle("DQN a function, episode: " + str(episode))
        else:
            plt.figure("a_1D", figsize=(19.2, 10.8), dpi=100)
            plt.suptitle("DQN a function, episode: " + str(episode))

        for index_state_req in range(n_state_req):
            plt.plot(state_res, a[index_state_req], c=colors[index_state_req])
        plt.xlabel('free servers')

        if do_not_save:
            plt.show()
        else:
            plt.figure("q_1D").savefig(self.plot_serie+"_q_"+str(episode)+".png")
            plt.figure("a_1D").savefig(self.plot_serie+"_a_"+str(episode)+".png")

        plt.close("all")
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

#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
        self.full = False

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1
        if self.pointer == self.capacity:
            self.pointer = 0
            self.full = True

    def samples(self, batch_size):
        if self.full:
            indices = np.random.choice(self.capacity, size=batch_size)
        else:
            indices = np.random.choice(self.pointer, size=batch_size)

        return self.data[indices, :]

####################################################
