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
# b_srv_1_reqsize_m
# b_srv_n_reqsize_m
# b_srv_n_reqsize_m_overlap
# b_srv_n_srvsize_o_reqsize_m
from b_srv_n_srvsize_o_reqsize_m import DataCenter

discount_rate = 0.9
learning_rate = 1.0E-3   # 0.001
epsilon       = 1.0E-1   # 0.1
n_steps       = int(6E4) # 40000
replace       = 1000
to_load_model = False
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
                   name_model='b_srv_n_srvsize_o_reqsize_m_DQN')

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

unfit_req = np.zeros((env.n_req_types))
spare_req = np.zeros((env.n_req_types))
discarded = np.zeros((env.n_req_types))
rejected  = np.zeros((env.n_req_types))
accepted  = np.zeros((env.n_req_types))
empty_slots = np.array([np.zeros(max(env.srv_size)+1) for s in range(env.n_server)])

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

    req_type  = env.state_req_type
    req_size  = s[0]
    num_empty_slots_at_server = s[1 : 1 + env.n_server]

    a = epsilon_greed_policy(epsilon, s)

    # alternative heuristic: use the less loaded server
    # server = np.argmax(num_empty_slots_at_server)
    # a = env.n_server
    # if req_size <= num_empty_slots_at_server[server]:
    #     a = server

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
        rwd_avg = 0.999 * rwd_avg + 0.001 * r
    else:
        rwd_avg = 0.999 * rwd_avg

    rwd_history.append(rwd_avg * suma_rwd)

    DQN.memory.store_transition(s, a, r, s_)

    if step > steps_to_start_learning and keep_learning:
        error = DQN.learn()
        error_avg = 0.999 * error_avg + 0.001 * error
        error_history.append(error_avg)

    if step % 1000 == 0:
        print('step: ', step,
              ', r_avg: %.3f' % (rwd_avg * suma_rwd))

    s = s_  # move to next state

print('r_avg = %.3f' % (rwd_avg * suma_rwd) )
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

DQN.save_model()

plot_history("reward", rwd_history, True)
plot_history("error", error_history, True)








