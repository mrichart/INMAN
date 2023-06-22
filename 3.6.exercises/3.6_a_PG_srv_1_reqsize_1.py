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
from b_srv_1_reqsize_1 import DataCenter

discount_rate = 0.99
learning_rate = 0.01   # 0.01
epsilon       = 0.00    # 0.1
n_steps       = 20000 # 200000
to_load_model = True
keep_learning = False
# it seems the learned policy is critical, the r_avg results in 2.25
# but when in choose_action(s) using return np.argmax(act_prob)
# we get the r_avg = 2.6 as the other algorithms.

env = DataCenter()

PG = PolicyGradient(
    n_features=env.n_features,
    n_actions=env.n_actions,
    learning_rate=learning_rate,
    reward_decay=discount_rate,
    to_load_model=to_load_model,
    name_model="b_srv_1_reqsize_1_PG")

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return PG.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

s = env.reset()

r_avg = 2.6 / sum(env.req_val)

discarded = np.zeros((env.n_req_types))
accepted  = np.zeros((env.n_req_types))
rejected  = np.zeros((env.n_req_types))
empty_slots = np.zeros((env.srv_size + 1))

for step in range(1, n_steps+1):

    req_type = s[0]
    num_empty_slots = s[1]

    a = epsilon_greed_policy(epsilon, s)

    empty_slots[num_empty_slots] += 1

    if a == 0:
        if num_empty_slots == 0:
            rejected[req_type] += 1
        else :
            accepted[req_type] += 1

    if a == 1:
        discarded[req_type] += 1

    s_, r = env.step(a)

    r_avg = 0.9999 * r_avg + 0.0001 * r

    PG.memory.store_transition(s, a, r)

    s = s_

    if step % 10 == 0:
        print('step: ', step,
              ', r_avg: %.3f' % (r_avg * sum(env.req_val)))
        if keep_learning:
            PG.learn()

print('r_avg = %.4f' % (r_avg * sum(env.req_val)) )
print('full  = %.4f' % (empty_slots[0] / n_steps))
print('discarded = %.4f' % (sum(discarded) / n_steps) )
print('rejected  = %.4f' % (sum(rejected) / n_steps) )
print('accepted  = %.4f' % (sum(accepted) / n_steps) )
print('dis_rate  = ', discarded / n_steps)
print('rej_rate  = ', rejected / n_steps)
print('acc_rate  = ', accepted / n_steps)
print('empty slots = ', empty_slots / n_steps)

PG.save_model()
PG.plot_a_2D_DataCenter("last_episode", True)

