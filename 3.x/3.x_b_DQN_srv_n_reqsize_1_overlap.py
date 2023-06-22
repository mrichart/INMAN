import sys
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=150)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x) if x%1 else "{0:0.0f}".format(x)})

sys.path.append('./util')
from DQN import DeepQNetwork

sys.path.append('./env')
from b_srv_n_reqsize_1_overlap import FacilityAssignment

discount_rate = 0.9
learning_rate = 0.001 # 0.001
epsilon       = 0.1   # 0.1
n_steps       = 10000 # 10000
replace       = 1000  # 1000
to_load_model = False
keep_learning = False

env = FacilityAssignment()

DQN = DeepQNetwork(n_features=env.n_features,
                   n_actions=env.n_actions,
                   learning_rate=learning_rate,
                   reward_decay=discount_rate,
                   memory_size=2**16,  # 65536
                   batch_size=128,  # 128
                   replace_target_iter=replace,
                   to_load_model=to_load_model,
                   name_model='b_srv_n_reqsize_1_overlap_DQN')

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return DQN.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

steps_to_start_learning = DQN.batch_size * 10

s = env.reset()

r_avg = 0.8
# r == -2 : out of range
# r == -1 : spare
# r ==  0 : discarded
# r ==  1 : accepted
r_account = np.zeros(4)

for step in range(1, n_steps+1):

    a = epsilon_greed_policy(epsilon, s)

    # alternative heuristic, use the less loaded if in range:
    # in_range = env.state_in_range
    # content = env.state_content
    # ordered_servers = np.argsort(content)
    # for server in ordered_servers:
    #     if in_range[server]:
    #         a = server
    #         break

    s_, r = env.step(a)

    r_account[r + 2] += 1 # r from [-2,1] -> r+2 from [0,3]
    r_avg = 0.999 * r_avg + 0.001 * r

    DQN.memory.store_transition(s, a, r, s_)

    if step > steps_to_start_learning and keep_learning:
        error = DQN.learn()

    s = s_  # move to next state

    if step % 100 == 0:
        print('step: ', step,
              ', r_avg: %.3f' % r_avg,
              ', r_account: ', r_account)

r_account = r_account / step
r_avg = (-2 * r_account[0]) + \
        (-1 * r_account[1]) + \
        (0 * r_account[2]) + \
        (1 * r_account[3])

print('r_account: ', r_account,
      ', r_avg: %.3f' % r_avg,
      ', acceptance_ratio: ', r_account[3])

DQN.save_model()








