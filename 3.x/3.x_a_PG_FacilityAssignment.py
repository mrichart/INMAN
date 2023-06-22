"""
Policy Gradient.
"""

import sys
import numpy as np
import numpy.random as rnd

np.set_printoptions(linewidth=150)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x) if x%1 else "{0:0.0f}".format(x)})

sys.path.append('../util')
from PG import PolicyGradient

sys.path.append('../env')
from a_facility_assignment import FacilityAssignment    # at PG we must use:
                                                        # * (1/(1-gamma))
                                                        # q -= np.mean(q)
                                                        # q /= np.std(q)

discount_rate = 0.9
learning_rate = 0.001 # 0.001
epsilon       = 0.1   # 0.1
n_episodes    = 2000  # 2000
to_load_model = True
keep_learning = True

env = FacilityAssignment()

PG = PolicyGradient(
    n_features=env.n_features,
    n_actions=env.n_actions,
    learning_rate=learning_rate,
    reward_decay=discount_rate,
    to_load_model=to_load_model,
    name_model="a_facility_assignment_PG")

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return PG.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

step      = 0
r_account = np.zeros(4)

for episode in range(1, n_episodes+1):

    s = env.reset()

    while True:

        step += 1

        a = epsilon_greed_policy(epsilon, s)

        s_, r, done = env.step(a)

        r_account[r + 2] += 1  # r from [-2,1] -> r+2 from [0,3]

        PG.memory.store_transition(s, a, r)

        s = s_

        if done:
            if episode % 100 == 0:
                print('episode: ', episode,
                      ', r_account: ', r_account)
            if keep_learning:
                PG.learn()
            break

r_account = r_account / step

print('r_account: ', r_account,
      ' acceptance_ratio: %.3f' % r_account[3])

PG.save_model()

