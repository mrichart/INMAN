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
from b_srv_n_srvsize_o_reqsize_m_keep_req import DataCenter

discount_rate = 0.9
learning_rate = 0.001 # 0.001
epsilon       = 0.1   # 0.1
n_steps       = 40000 # 40000
replace       = 1000
to_load_model = True
keep_learning = True

env = DataCenter()

DQN = DeepQNetwork(n_features=env.n_features,
                   n_actions=env.n_actions,
                   learning_rate=learning_rate,
                   reward_decay=discount_rate,
                   memory_size=2**16,
                   batch_size =128,
                   replace_target_iter=replace,
                   to_load_model=to_load_model,
                   name_model='b_srv_n_srvsize_o_reqsize_m_keep_req_DQN')

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return DQN.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

steps_to_start_learning = DQN.batch_size * 10

s = env.reset()

suma_rwd = sum(env.req_val)
rwd_avg = 0.0 / suma_rwd
rwd_history = []
error_avg = 0.0
error_history = []

r_avg       = 0
pend_reqs   = np.zeros((env.n_req_types))
empty_slots = np.array([np.zeros(env.srv_size[s]+1) for s in range(env.n_server)])
ordered_req = np.flip(np.argsort(env.req_size), axis=0)

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

for step in range(1, n_steps+1):

    num_empty_slots_at_server = s[env.n_req_types: env.n_req_types + env.n_server]

    a = epsilon_greed_policy(epsilon, s)

    # alternative heuristic: allocate bigger pend_req in server most free
    # a = 9
    # server = np.argmax(num_empty_slots_at_server)
    # for req in ordered_req:
    #     if env.state_pend_reqs[req] > 0 and num_empty_slots_at_server[server] > env.req_size[req]:
    #         a = env.inv_act_map([req, server])
    #         break

    for server in range(env.n_server):
        empty_slots[server][num_empty_slots_at_server[server]] += 1

    pend_reqs += env.state_pend_reqs

    s_, r = env.step(a)

    if r >= 0:
        r_avg = 0.999 * r_avg + 0.001 * r
    else:
        r_avg = 0.999 * r_avg

    rwd_history.append(r_avg * suma_rwd)

    if r> 0:
        rwd_avg += r

    DQN.memory.store_transition(s, a, r, s_)

    if step > steps_to_start_learning and keep_learning:
        error = DQN.learn()
        error_avg = 0.999 * error_avg + 0.001 * error
        error_history.append(error_avg)

    if step % 1000 == 0:
        print('step: ', step,
              ', r_avg: %.3f' % (r_avg * sum(env.req_val)),
              ', rwd_avg: %.3f' % ((rwd_avg * sum(env.req_val)) / 1000),
              ', pend_reqs: ', pend_reqs / 1000,
              ', entered: ', env.entered, ', exited: ', env.exited,
              ', last step pend_reqs: ', env.state_pend_reqs)
        pend_reqs = np.zeros((env.n_req_types))
        rwd_avg = 0
        s_ = env.reset()

    s = s_  # move to next state

print('r_avg = %.3f' % (r_avg * sum(env.req_val)) )
print('empty slots:\n', empty_slots / n_steps)

DQN.save_model()

plot_history("reward", rwd_history, True)
plot_history("error", error_history, True)








