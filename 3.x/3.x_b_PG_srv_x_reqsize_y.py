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
# b_srv_1_reqsize_m
# b_srv_n_reqsize_m
# b_srv_n_reqsize_m_overlap
# b_srv_n_srvsize_o_reqsize_m
from b_srv_n_srvsize_o_reqsize_m import DataCenter

discount_rate = 0.9
learning_rate = 0.001  # 0.001
epsilon       = 0.0    # 0.1
n_steps       = 200000 # 150000
to_load_model = True
keep_learning = True

env = DataCenter()

PG = PolicyGradient(
    n_features=env.n_features,
    n_actions=env.n_actions,
    learning_rate=learning_rate,
    reward_decay=discount_rate,
    to_load_model=to_load_model,
    name_model="b_srv_n_srvsize_o_reqsize_m_PG")

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return PG.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

s = env.reset()

r_avg     = 0
unfit_req = np.zeros((env.n_req_types))
spare_req = np.zeros((env.n_req_types))
discarded = np.zeros((env.n_req_types))
rejected  = np.zeros((env.n_req_types))
accepted  = np.zeros((env.n_req_types))
#empty_slots = np.array([np.zeros(env.srv_size + 1) for s in range(env.n_server)])
empty_slots = np.array([np.zeros(env.srv_size[s]+1) for s in range(env.n_server)])

for step in range(1, n_steps+1):

    req_type  = env.state_req_type
    req_size  = s[0]
    num_empty_slots_at_server = s[1:env.n_server + 1]

    a = epsilon_greed_policy(epsilon, s)

    for server in range(env.n_server):
        empty_slots[server][num_empty_slots_at_server[server]] += 1

    if req_size > max(num_empty_slots_at_server):
        unfit_req[req_type] += 1

    if a < env.n_server:
        if req_size > num_empty_slots_at_server[a]:
            rejected[req_type] += 1
        else:
            accepted[req_type] += 1

    if a == env.n_server:
        discarded[req_type] += 1
        if req_size <= max(num_empty_slots_at_server):
            spare_req[req_type] += 1

    s_, r = env.step(a)

    if r >= 0:
        r_avg = 0.999 * r_avg + 0.001 * r
    else:
        r_avg = 0.999 * r_avg

    PG.memory.store_transition(s, a, r)

    s = s_

    if step % 100 == 0:
        print('step: ', step,
              ', r_avg: %.3f' % (r_avg * sum(env.req_val)))
        if keep_learning:
            PG.learn()

print('r_avg = %.3f' % (r_avg * sum(env.req_val)) )
print('unfit_req = %.3f' % (sum(unfit_req) / n_steps) )
print('spare_req = %.3f' % (sum(spare_req) / n_steps) )
print('discarded = %.3f' % (sum(discarded) / n_steps) )
print('rejected  = %.3f' % (sum(rejected) / n_steps) )
print('accepted  = %.3f' % (sum(accepted) / n_steps) )
print('unfit_req = ', unfit_req / n_steps)
print('spare_req = ', spare_req / n_steps)
print('dis_rate  = ', discarded / n_steps)
print('rej_rate  = ', rejected / n_steps)
print('acc_rate  = ', accepted / n_steps)
print('empty slots:\n', empty_slots / n_steps)

PG.save_model()

