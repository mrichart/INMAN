import sys
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=150)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x) if x%1 else "{0:0.0f}".format(x)})

sys.path.append('../util')
from DQN import DeepQNetwork

sys.path.append('../env')
from b_srv_n_srvsize_o_reqsize_m_DC_master import DC_master

discount_rate = 0.9
learning_rate = 0.001 # 0.001
epsilon       = 0.1   # 0.1
n_steps       = 40000 # 40000
replace       = 1000
to_load_model = True
keep_learning = True

env = DC_master(to_load_model, keep_learning)

DQN = DeepQNetwork(n_features=env.n_features,
                       n_actions=env.n_actions,
                       learning_rate=learning_rate,
                       reward_decay=discount_rate,
                       memory_size=2**16,
                       batch_size =128,
                       replace_target_iter=replace,
                       to_load_model=to_load_model,
                       name_model='b_srv_n_srvsize_o_reqsize_m_DCs_'+'master'+'_DQN',
                       id='master')

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return DQN.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

steps_to_start_learning = DQN.batch_size * 10

s = env.reset()

r_avg = 0

for step in range(1, n_steps+1):

    a = epsilon_greed_policy(epsilon, s)

    s_, r = env.step(a)

    if r >= 0:
        r_avg = 0.999 * r_avg + 0.001 * r
    else:
        r_avg = 0.999 * r_avg

    DQN.memory.store_transition(s, a, r, s_)

    if step > steps_to_start_learning and keep_learning:
        error = DQN.learn()

    if step % 1000 == 0:
        print('step: ', step,
              ', r_avg: %.3f' % (r_avg * sum(env.env_DC[0].req_val)))

    s = s_

print('r_avg = %.3f' % (r_avg * sum(env.env_DC[0].req_val)) )

for dc in range(env.n_DCs):
    env.DQN_DC[dc].save_model()

DQN.save_model()








