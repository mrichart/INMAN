import sys
import numpy as np
import numpy.random as rnd

import datetime

sys.path.append('./util')
from DQN import DeepQNetwork
from DynamicArray import DynamicArray
from ResetEpisodes import ResetEpisodes

sys.path.append('./env')
from b_srv_1_reqtime_m import DataCenter

class runDataCenter_DQN:

    def __init__(self,
                 max_num_episodes       = 50,
                 max_steps_in_episode   = 2000,
                 to_load_model          = True,
                 keep_learning          = True,
                 learning_rate          = 0.001,
                 reward_decay           = 0.995,
                 max_epsilon            = 0.1, # 0.50,
                 min_epsilon            = 0.1, # 0.05,
                 replace                = 400,
                 profile                = 0,
                 ):

        self.max_num_episodes       = max_num_episodes
        self.max_steps_in_episode   = max_steps_in_episode
        self.to_load_model          = to_load_model
        self.keep_learning          = keep_learning
        self.learning_rate          = learning_rate
        self.reward_decay           = reward_decay
        self.max_epsilon            = max_epsilon
        self.min_epsilon            = min_epsilon
        self.epsilon_step           = (max_epsilon-min_epsilon) / (max_steps_in_episode / 2)
        self.replace                = replace
        self.profile                = profile

        self.env = DataCenter(n_actions=3 + 1, profile=self.profile)

        self.case  = "_replace_" + str(self.replace)
        self.case += "_lr_"      + str(self.learning_rate)
        self.case += "_gamma_"   + str(self.reward_decay)
        self.case += "_profile_" + str(self.profile)
        self.case += "_load_"    + str(self.env.load)

        self.DQN = DeepQNetwork(n_features=self.env.n_features,
                           n_actions=self.env.n_actions,
                           learning_rate=self.learning_rate,
                           reward_decay=self.reward_decay,
                           memory_size=2 ** 16,  # 65536
                           batch_size=2 ** 7,  # 128
                           replace_target_iter=self.replace,
                           to_load_model=self.to_load_model,
                           name_model='b_srv_1_reqtime_m_DQN'+self.case)

    def epsilon_greed_policy(self, epsilon, s):

        if epsilon > self.min_epsilon:
            epsilon -= self.epsilon_step

        if rnd.random() > epsilon:
            return self.DQN.choose_action(s)
        else:
            return rnd.choice(self.env.n_actions)

    def run(self):

        env = self.env
        DQN = self.DQN

        steps_to_start_learning = DQN.batch_size * 10

        s = env.reset()

        save_model_every  = 100 # episodes

        total_steps = 0

        CT_episode_series = DynamicArray()
        SD_episode_series = DynamicArray()
        r_episode_series  = DynamicArray()
        acts_total        = np.zeros(env.n_actions, dtype=int)
        reset_episodes    = ResetEpisodes(self.max_num_episodes)

        for episode in range(self.max_num_episodes):

            epsilon = self.max_epsilon
            steps_in_episode  = 0
            reqs_in_episode   = 0
            CT_sum_in_episode = 0
            SD_sum_in_episode = 0
            r_sum_in_episode  = 0
            acts_in_episode   = np.zeros(env.n_actions, dtype=int)

            while True:

                a = self.epsilon_greed_policy(epsilon, s)

                #a = self.env.SJF()

                s_, r, done, new_req, success = env.step(a)

                total_steps        += 1

                steps_in_episode   += 1
                reqs_in_episode    += new_req
                CT_sum_in_episode  += env.inside_jobs() # said Completion Time, but inside_jobs in fact
                SD_sum_in_episode  += env.slowdown()    # said Slow Down, but relative completion time in fact
                r_sum_in_episode   += r                 # reward, used relative waiting time
                acts_in_episode[a] += 1
                acts_total[a]      += 1

                DQN.memory.store_transition(s, a, r, s_)

                s = s_

                if total_steps > steps_to_start_learning and self.keep_learning:
                    error = DQN.learn()

                if steps_in_episode == self.max_steps_in_episode:
                    done = True

                if done:

                    if steps_in_episode < self.max_steps_in_episode:
                        s = env.reset()
                        reset_episodes.add(episode)

                    CT_episode_series.append([CT_sum_in_episode, reqs_in_episode])
                    SD_episode_series.append([SD_sum_in_episode, reqs_in_episode])
                    r_episode_series.append ([-r_sum_in_episode, reqs_in_episode]) # steps_in_episode

                    print(' Episode: ', episode,
                          '\t| CT_in_episode: %.4f' % (CT_sum_in_episode / reqs_in_episode),
                          '\t| SD_in_episode: %.4f' % (SD_sum_in_episode / reqs_in_episode),
                          '\t| r_in_episode:  %.4f' % (-r_sum_in_episode / reqs_in_episode), # steps_in_episode
                          '\t| steps_in_episode: ', steps_in_episode,
                          '\t| a =', ' '.join(str([i, acts_in_episode[i]]) for i in range(env.n_actions)),
                          )

                    if episode % 100 == 0 and episode != 0:
                        print(
                            # '\t| CT_avg: ', round(CT_episode_series.avg_tillN_in_window(episode,100)(), 4),
                            '\t| SD_avg: ', round(SD_episode_series.avg_tillN_in_window(episode, 100), 4),
                            '\t| r_avg: ', round(r_episode_series.avg_tillN_in_window(episode, 100), 4),
                        )

                    if episode % save_model_every == 0 and episode != 0:
                        DQN.save_model()

                    break

        print(#'\t| CT_avg: ', round(CT_episode_series.avg_all(), 4),
              '\t| SD_avg: %.4f' % SD_episode_series.avg_all(),
              '\t| r_avg:  %.4f' %  r_episode_series.avg_all(),
              )
        print('\nacts_total =', ' '.join(str([i, acts_total[i]]) for i in range(env.n_actions)))
        print("\nreset_episodes: ", reset_episodes.show())

        DQN.save_model()

if __name__ == "__main__":

    print("starting time simulation: ", datetime.datetime.now(), '\n')
    runDataCenter = runDataCenter_DQN()
    runDataCenter.run()
    print("\nending time simulation: ", datetime.datetime.now())
