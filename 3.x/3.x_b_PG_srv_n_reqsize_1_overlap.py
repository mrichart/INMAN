"""
Policy Gradient.
"""

import sys
import numpy as np
import numpy.random as rnd

np.set_printoptions(linewidth=150)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x) if x%1 else "{0:0.0f}".format(x)})

sys.path.append('./util')
from PG import PolicyGradient

sys.path.append('./env')
from b_srv_n_reqsize_1_overlap import FacilityAssignment

discount_rate = 0.9
learning_rate = 0.001  # 0.001
epsilon       = 0.0    # 0.1
n_steps       = 15000 # 150000
to_load_model = False
keep_learning = True

env = FacilityAssignment()

PG = PolicyGradient(
    n_features=env.n_features,
    n_actions=env.n_actions,
    learning_rate=learning_rate,
    reward_decay=discount_rate,
    to_load_model=to_load_model,
    name_model="b_srv_n_reqsize_1_overlap_PG")

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return PG.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

s = env.reset()

r_avg = 0.8
r_account = np.zeros(4)

for step in range(1, n_steps+1):

    a = epsilon_greed_policy(epsilon, s)

    s_, r = env.step(a)

    r_account[r + 2] += 1  # r from [-2,1] -> r+2 from [0,3]
    r_avg = 0.999 * r_avg + 0.001 * r

    PG.memory.store_transition(s, a, r)

    s = s_

    if step % 100 == 0:
        print('step: ', step,
              ', r_avg: %.3f' % r_avg,
              ', r_account: ', r_account)
        if keep_learning:
            PG.learn()

r_account = r_account / step
r_avg = (-2 * r_account[0]) + \
        (-1 * r_account[1]) + \
        (0 * r_account[2]) + \
        (1 * r_account[3])

print('r_account: ', r_account,
      ', r_avg: %.3f' % r_avg,
      ', acceptance_ratio: ', r_account[3])

PG.save_model()

