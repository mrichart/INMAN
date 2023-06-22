import sys
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=150)

sys.path.append('../util')
from DQN import DeepQNetwork

sys.path.append('../env')
from a_knapsack_v2 import Knapsack

discount_rate = 0.9
learning_rate = 0.001 # 0.01 or 0.001
epsilon       = 0.1 # 0.1
n_episodes    = 2048
replace       = 128
to_load_model = False
keep_learning = True

env = Knapsack()

# it oscillates regarding the tracking of the optimal policy.

DQN = DeepQNetwork(n_features=env.n_features,
                   n_actions=env.n_actions,
                   learning_rate=learning_rate,
                   reward_decay=discount_rate,
                   memory_size=64,
                   batch_size=4,
                   replace_target_iter=replace,
                   to_load_model=to_load_model,
                   name_model='a_knapsack_v2_DQN')

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return DQN.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

steps_to_start_learning = DQN.batch_size * 10
step = 0
visited_sars_ = []

print('req_size: ', env.req_size)
print('req_val:  ', env.req_val)
print('')

for episode in range(n_episodes):

    s = env.reset()
    R = 0
    acts = []

    while True:

        step += 1
        a = epsilon_greed_policy(epsilon, s)
        acts.append(env.req_size[a])
        s_, r, done = env.step(a)
        R += r

        new_elem = np.hstack((s, a, r, s_))
        if not any([(new_elem == elem).all() for elem in visited_sars_]):
            visited_sars_.append(new_elem)
            #print('entries: ', len(visited_sars_))
            DQN.memory.store_transition(s, a, r, s_)

        if step > steps_to_start_learning and keep_learning:
            error = DQN.learn()

        s = s_  # move to next state

        if done:
            if episode % 10 == 0:
                print("content: ", env.state_content,
                      ', R: %.1f' % (R*sum(env.req_val)),
                      ', acts: ', acts)
            break

DQN.save_model()








