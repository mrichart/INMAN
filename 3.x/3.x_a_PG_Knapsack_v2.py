"""
Policy Gradient.
"""

import sys
import numpy as np
import numpy.random as rnd

np.set_printoptions(linewidth=150)

sys.path.append('./util')
from PG import PolicyGradient

sys.path.append('./env')
from a_knapsack_v2 import Knapsack # at PG we must use:
                                   # * (1/(1-gamma))
                                   # q -= np.mean(q)
                                   # q /= np.std(q)

discount_rate = 0.9
learning_rate = 0.001 # 0.01 or 0.001
epsilon       = 0.1 # 0.1
n_episodes    = 20000
to_load_model = False
keep_learning = True

env = Knapsack()

PG = PolicyGradient(
    n_features=env.n_features,
    n_actions=env.n_actions,
    learning_rate=learning_rate,
    reward_decay=discount_rate,
    to_load_model=to_load_model,
    name_model="a_knapsack_v2_PG")

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return PG.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

step = 0

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

        PG.memory.store_transition(s, a, r)

        s = s_

        if done:
            if episode % 10 == 0:
                print("content: ", env.state_content,
                      ', R: %.1f' % (R*sum(env.req_val)),
                      ', acts: ', acts)
            if keep_learning:
                PG.learn()
            break

PG.save_model()

