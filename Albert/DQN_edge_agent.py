"""
Deep_Q_Network.
"""
import numpy as np
import sys
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import matplotlib.pyplot as plt

sys.path.append('../util')
from DQN_NN import NN_Q_s

sys.path.append('../env')
from b_srv_1_reqsize_1 import DataCenter

from DQN import DeepQNetwork

class DeepQNetwork_edge:

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
            another_model = None,
            session = None
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
        self.id = id

        self.s = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.s_ = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.q_target = tf.placeholder(tf.float32, shape=[None, self.n_actions])

        self.q_s = NN_Q_s(self.s, self.n_actions, scope='eval_net_params' + id)  # (None, n_actions)
        self.q_s_ = NN_Q_s(self.s_, self.n_actions, scope='target_net_params' + id)  # (None, n_actions)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net_params' + id)
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net_params' + id)

        self.tmp = tf.squared_difference(self.q_target, self.q_s)  # (None, n_actions)
        self.error = self.n_actions * tf.reduce_mean(
            self.tmp)  # (1, ) (n_actions-1) are zeros in each sub-array for this mean.
        # self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.error)

        self.OptC = tf.train.RMSPropOptimizer(self.lr)
        self.c_grads = tf.gradients(self.error, self.e_params)  # valor del gradient que s'aplicara
        self.train_op = self.OptC.apply_gradients(zip(self.c_grads, self.e_params))

        self.replace_target_op = [t.assign(e) for t, e in zip(self.t_params, self.e_params)]
        self.cp_another_model = [tf.assign(t, e) for t, e in zip(self.e_params, another_model)]

        self.sess = session
        self.saver = tf.train.Saver()

        if to_load_model:
            self.load_model()

        self.error_history = []

    def get_model_weights(self):
        return self.sess.run(self.e_params)

    def update_model(self, model_weights):
        self.sess.run([tf.assign(e, w) for e, w in zip(self.e_params, model_weights)])

    def update_from_another_model(self):
        self.sess.run(self.cp_another_model)

    def choose_action(self, s):
        actions_value = self.sess.run(self.q_s, feed_dict={self.s: s[np.newaxis, :]})[0]
        return np.argmax(actions_value)

    def learn(self):
        samples = self.memory.samples(self.batch_size)
        q_s = self.sess.run(self.q_s, {self.s: samples[:, :self.n_features]})  # (None, n_actions)
        q_s_ = self.sess.run(self.q_s_, {self.s_: samples[:, -self.n_features:]})  # (None, n_actions)
        q_target = q_s.copy()  # (None, n_actions)

        samples_indices = np.arange(self.batch_size, dtype=np.int32)
        actions = samples[:, self.n_features].astype(int)  # (None, )
        rewards = samples[:, self.n_features + 1]  # (None, )
        tmp1 = np.max(q_s_, axis=1)  # (None, )
        tmp2 = self.reward_decay * tmp1  # (None, )
        tmp3 = rewards + tmp2  # (None, )
        q_target[samples_indices, actions] = tmp3

        feed_dict = {self.s: samples[:, :self.n_features], self.q_target: q_target}
        _, c_grads, error, tmp = self.sess.run([self.train_op, self.c_grads, self.error, self.tmp], feed_dict)

        # self.error_history.append(error)

        self.replace_counter += 1
        if self.replace_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        return np.sqrt(error)

    def save_model(self):
        save_path = self.saver.save(self.sess, "../model/" + self.name_model + ".ckpt")
        # print("Model saved with path: %s" % save_path)

    def load_model(self):
        self.saver.restore(self.sess, "../model/" + self.name_model + ".ckpt")
        # print("Model loaded from: ", self.dir + "/output/" + self.name_model + ".ckpt")

    def reset_model(self):
        self.sess.run(tf.global_variables_initializer())
        self.error_history = []


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

env = DataCenter()

my_session = tf.Session()

dqn1 = DeepQNetwork(n_features=env.n_features,
                    n_actions=env.n_actions,
                    id = "1",
                    session = my_session)

dqn2 = DeepQNetwork_edge(n_features=env.n_features,
                         n_actions=env.n_actions,
                         id = "2",
                         another_model=dqn1.get_model(),
                         session= my_session)

dqn3 = DeepQNetwork(n_features=env.n_features,
                    n_actions=env.n_actions,
                    id = "3",
                    session= my_session)

my_session.run(tf.global_variables_initializer())

model1_weights = dqn1.get_model_weights()
model2_weights = dqn2.get_model_weights()
model3_weights = dqn3.get_model_weights()

dqn2.update_model(model1_weights)
model4_val = dqn2.get_model_weights()

dqn2.update_model(model3_weights)
model5_val = dqn2.get_model_weights()

dqn2.update_from_another_model()
model6_val = dqn2.get_model_weights()

print("end")

