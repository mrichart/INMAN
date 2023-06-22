import sys
import datetime
import numpy as np
import threading
import tensorflow as tf

sys.path.append('../util')
from GlobalEpisode import GlobalEpisode
from A3C_dis import A3C_global
from A3C_dis import A3C

sys.path.append('../env')
from c_srv_1_reqsize_cont_act_cont import DataCenter

class runDataCenter_A3C:

    def __init__(self,
                 seed                   = 1,
                 max_num_episodes       = 1000,
                 max_steps_in_episode   = 20000,
                 to_load_model          = False,
                 keep_learning          = True,
                 learning_rate          = 0.001,
                 reward_decay           = 0.99,
                 max_epsilon            = 0.50,
                 min_epsilon            = 0.05,
                 ahead                  = 1,
                 n_workers              = 8,
                 ):

        self.seed                   = seed
        self.max_num_episodes       = max_num_episodes
        self.max_steps_in_episode   = max_steps_in_episode
        self.to_load_model          = to_load_model
        self.keep_learning          = keep_learning
        self.learning_rate          = learning_rate
        self.reward_decay           = reward_decay
        self.max_epsilon            = max_epsilon
        self.min_epsilon            = min_epsilon
        self.AHEAD                  = ahead
        self.N_WORKERS              = n_workers

        self.save_model_every = 100  # episodes

        env = DataCenter()
        self.env = env

        self.case  = "_seed_"      + str(self.seed)
        self.case += "_AHEAD_"     + str(self.AHEAD)
        self.case += "_N_WORKERS_" + str(self.N_WORKERS)
        self.case += "_lr_"        + str(self.learning_rate)
        self.case += "_gamma_"     + str(self.reward_decay)

        self.globalMLT = A3C_global(seed=self.seed,
                               name_model="A3C"+self.case,
                               scope="global",
                               sess=tf.Session(),
                               n_features=env.n_features,
                               n_actions=env.n_actions,
                               lr_C=self.learning_rate,
                               lr_A=self.learning_rate/10.0,
                               reward_decay=self.reward_decay)

        self.gE = GlobalEpisode(self)

        self.workers = []
        for i in range(self.N_WORKERS):
            i_name = 'W_%i' % i
            self.workers.append(Worker(i_name, self))

        self.saver = tf.train.Saver()
        self.gE.setSaver(self.saver)

        if self.to_load_model:
            self.globalMLT.load_model(self.saver)
        else:
            self.globalMLT.sess.run(tf.global_variables_initializer())

    def run(self):

        COORD = tf.train.Coordinator()
        worker_threads = []
        for worker in self.workers:
            t = threading.Thread(target=worker.work)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)

        print('\t| r_avg: ', round(self.gE.r_episode_series.avg_all(), 4))
        print('\nacts_total =', ' '.join(str([i, self.gE.acts_total[i]]) for i in range(self.env.n_actions)))

        supercase  = "learn_"  + str(self.keep_learning) + datetime.datetime.now().strftime("_%Y-%m-%d-%Hh%Mm")
        supercase += "_epis_"  + str(self.max_num_episodes)
        supercase += "_steps_" + str(self.max_steps_in_episode)

        self.globalMLT.save_model(self.saver)
        self.workers[0].AC.plot_graph(self.workers[0].env.history[:, 0:4], supercase + "_assig_vs_req_", do_not_save=False)
        self.workers[0].AC.plot_graph(self.workers[0].env.history[:, 4], supercase + "_sla_", do_not_save=False)
        self.workers[0].AC.plot_graph(self.workers[0].env.history[:, 5], supercase + "_used_", do_not_save=False)
        self.workers[0].AC.plot_graph(self.gE.r_episode_series.all_elements(), supercase + "__episodes__r_", do_not_save=False)
        self.workers[0].AC.plot_graph(self.gE.r_episode_series.cumulative_avg_all(), supercase + "__avg_all__r__", do_not_save=False)

class Worker:

    def __init__(self, name, runDataCenter_A3C):

        self.rDC = runDataCenter_A3C
        self.name = name

        self.env = DataCenter()
        self.AC  = A3C(scope=name, gAC=self.rDC.globalMLT)

        self.gE = self.rDC.gE

    def work(self):

        env = self.env
        AC  = self.AC

        max_epsilon = self.rDC.max_epsilon
        min_epsilon = self.rDC.min_epsilon
        epsilon_step = (max_epsilon-min_epsilon) / (self.rDC.max_steps_in_episode / 2)

        r_avg = 0

        s = env.reset()
        steps_ahead = 0  # HAS BEEN MOVED HERE!!!

        done = False

        for episode in range(int(self.rDC.max_num_episodes/self.rDC.N_WORKERS)):

            epsilon = max_epsilon
            steps_in_episode    = 0
            r_sum_in_episode    = 0
            acts_in_episode     = np.zeros(env.n_actions, dtype=int)

            while True :

                if np.random.uniform(0,1) < epsilon:
                    a = np.random.randint(0, env.n_actions)
                else:
                    a = AC.choose_action(s)

                if epsilon > min_epsilon:
                    epsilon -= epsilon_step

                if np.isnan(a):
                    print("Worker:", self.name,
                          "nan at episode: ", episode,
                          " the_step: ", steps_in_episode)
                    break

                s_, r = env.step(a)

                steps_ahead         += 1

                steps_in_episode    += 1
                r_sum_in_episode    += r
                acts_in_episode[a]  += 1

                AC.memoryAhead.store_transition(s, a, r)
                #AC.memoryAhead.store_transition(s, a, r - r_avg)

                if (steps_ahead % self.rDC.AHEAD == 0 or done) and self.rDC.keep_learning:
                    v_s_ = AC.v_s(s_)
                    if done: v_s_ = 0.0
                    error = AC.learn(v_s_)
                    steps_ahead = 0

                s = s_

                if steps_in_episode == self.rDC.max_steps_in_episode:
                    done = True

                if done:

                    globalEpi, r_avg = self.gE.done(r_sum_in_episode,
                                                    steps_in_episode,
                                                    acts_in_episode)
                    '''
                    print(' Worker: ', self.name,
                          '\t| Episode: ', globalEpi,
                          '\t| r_in_episode: ',  round(r_sum_in_episode / steps_in_episode, 4),
                          '\t| a =', ' '.join(str([i, acts_in_episode[i]]) for i in range(env.n_actions)),
                          )
                    '''
                    break

if __name__ == "__main__":

    print("starting time simulation: ", datetime.datetime.now(), '\n')
    runDataCenter = runDataCenter_A3C()
    runDataCenter.run()
    print("\nending time simulation: ", datetime.datetime.now())