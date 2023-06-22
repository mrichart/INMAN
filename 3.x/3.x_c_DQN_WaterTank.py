import sys
import numpy as np
import numpy.random as rnd

sys.path.append('../util')
from DQN import DeepQNetwork

sys.path.append('../env')
from watertank import WaterTank

discount_rate = 0.99
learning_rate = 0.01  # first 0.01, then 0.001
epsilon       = 0.3   # first 0.3,  then 0.1
n_episodes    = 10
n_steps       = 3000
replace       = 3000
to_load_model = False
keep_learning = False

env = WaterTank()

DQN = DeepQNetwork(n_features=env.n_features,
                   n_actions=env.n_actions,
                   learning_rate=learning_rate,
                   reward_decay=discount_rate,
                   memory_size=20048,
                   batch_size =2 ** 7,  # 128
                   replace_target_iter=replace,
                   to_load_model=to_load_model,
                   name_model='waterTank_DQN')

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return DQN.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

steps_to_start_learning = DQN.batch_size * 10

water_levels = np.zeros(n_episodes * n_steps, dtype=float)
rewards      = np.zeros(n_episodes * n_steps, dtype=float)

for episode in range(n_episodes):

    s = env.reset()

    for step in range(n_steps):

        a = epsilon_greed_policy(epsilon, s)

        # obvious policy, better than learned:
        # if s[0] < 5.0:
        #     a = 1
        # else:
        #     a = 0

        s_, r = env.step(a)

        water_levels.put(episode * n_steps + step, s_[0])
        rewards.put(episode * n_steps + step, r)

        DQN.memory.store_transition(s, a, r, s_)
        if step > steps_to_start_learning and keep_learning:
            error = DQN.learn()

        s = s_  # move to next state

    pointer = (episode + 1) * n_steps # pointer on concluding an episode
    print('r_avg: %.6f' % (np.sum(rewards[pointer - 100:pointer]) / 100))

DQN.save_model()

DQN.plot_data(water_levels, "Water Level", do_not_save=True)
DQN.plot_data(rewards, "Reward", do_not_save=True)







