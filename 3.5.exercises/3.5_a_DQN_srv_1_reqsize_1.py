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
from b_srv_1_reqsize_1 import DataCenter

discount_rate = 0.99
learning_rate = 1.0E-2   # 0.01   | to check:
epsilon       = 0.1E-1   # 0.1    | 0.0
n_steps       = int(2E4) # 60000  | 40000
replace       = 1000
to_load_model = False
keep_learning = True

env = DataCenter()

DQN = DeepQNetwork(n_features=env.n_features,
                   n_actions=env.n_actions,
                   learning_rate=learning_rate,
                   reward_decay=discount_rate,
                   memory_size=2 ** 15,  # (s,a,r,s_): (4x11)x2x5x(4x11) = 19360 entries
                   batch_size =2 ** 7,  # 128
                   replace_target_iter=replace,
                   to_load_model=to_load_model,
                   name_model='b_srv_1_reqsize_1_DQN')

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return DQN.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

steps_to_start_learning = DQN.batch_size * 10

s = env.reset()

suma_rwd = sum(env.req_val)
rwd_avg = 0 #2.63 / suma_rwd
rwd_history = []
error_avg = 0 # 0.19
error_history = []

discarded = np.zeros((env.n_req_types))
rejected  = np.zeros((env.n_req_types))
accepted  = np.zeros((env.n_req_types))
empty_slots  = np.zeros((env.srv_size + 1))

def plot_history(title, history, do_not_save):
    if do_not_save:
        plt.figure(title)
    else:
        plt.figure(title, figsize=(19.2, 10.8), dpi=100)
    plt.plot(range(len(history)), history)
    plt.ylabel(title)
    plt.xlabel('training steps')
    plt.grid()
    if do_not_save:
        plt.show()
    else:
        plt.figure(title).savefig(title+"_history.png")

    plt.close("all")

for step in range(n_steps):

    req_type = s[0]
    num_empty_slots = s[1]

    #epsilon = 0.99 * epsilon + 1.0E-2
    a = epsilon_greed_policy(epsilon, s)

    # do not discard any request:
    #a = 0

    # alternative heuristic, close to optimal policy:
    # if (req_type == 0 or req_type == 1):
    #     a = 0
    # else:
    #     a = 1

    empty_slots[num_empty_slots] += 1

    if a == 0:
        if num_empty_slots == 0:
            rejected[req_type] += 1
        else:
            accepted[req_type] += 1

    if a == 1:
        discarded[req_type] += 1

    s_, r = env.step(a)

    rwd_avg = 0.999 * rwd_avg + 0.001 * r
    rwd_history.append(rwd_avg*suma_rwd)

    DQN.memory.store_transition(s, a, r, s_)

    if step > steps_to_start_learning and keep_learning:
        error = DQN.learn()
        error_avg = 0.999 * error_avg + 0.001 * error
        error_history.append(error_avg)

    if step % 1000 == 0:
        print('step: ', step,
              ', r_avg: %.3f' % (rwd_avg * sum(env.req_val)))

    s = s_  # move to next state

DQN.save_model()

print('r_avg = %.4f' % (rwd_avg * sum(env.req_val)) )
print('full  = %.4f' % (empty_slots[0] / n_steps))
print('discarded = %.4f' % (sum(discarded) / n_steps) )
print('rejected  = %.4f' % (sum(rejected) / n_steps) )
print('accepted  = %.4f' % (sum(accepted) / n_steps) )
print('dis_rate  = ', discarded / n_steps)
print('rej_rate  = ', rejected / n_steps)
print('acc_rate  = ', accepted / n_steps)
print('empty slots = ', empty_slots / n_steps)

plot_history("reward", rwd_history, True)
plot_history("error", error_history, True)
DQN.plot_a_2D_DataCenter("all", True)
DQN.plot_q_a_1D_DataCenter("all", True)







