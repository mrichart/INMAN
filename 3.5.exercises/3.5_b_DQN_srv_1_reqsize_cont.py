import sys
import numpy.random as rnd

sys.path.append('../util')
from DQN import DeepQNetwork

sys.path.append('../env')
from c_srv_1_reqsize_cont import DataCenter

env = DataCenter()

discount_rate = 0.95
learning_rate = 0.01  # first 0.01, then 0.001
epsilon       = 0.1   # 0.1
n_steps       = 20000
replace       = env.one_cycle
to_load_model = False
keep_learning = True

DQN = DeepQNetwork(n_features=env.n_features,
                   n_actions=env.n_actions,
                   learning_rate=learning_rate,
                   reward_decay=discount_rate,
                   memory_size=2 ** 16,  # 65536
                   batch_size=2 ** 7,  # 128
                   replace_target_iter=replace,
                   to_load_model=to_load_model,
                   name_model='c_srv_1_reqsize_cont_DQN')

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return DQN.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

steps_to_start_learning = DQN.batch_size * 10

s = env.reset()
r_avg = 0

for step in range(n_steps):

    a = epsilon_greed_policy(epsilon, s)
    s_, r = env.step(a)
    r_avg = r_avg * 0.999 + r * 0.001
    DQN.memory.store_transition(s, a, r, s_)
    if step > steps_to_start_learning and keep_learning:
        error = DQN.learn()
    if step % env.one_cycle == 0:
        print('step: ', step,
              ', r_avg: %.4f' % r_avg)
    if r_avg > 0.995:
        break
    s = s_

DQN.save_model()

print('r_avg: %.4f' % r_avg)

DQN.plot_data(env.hist[:, 0:4], "Assig_vs_Req", do_not_save=True,
               labels = ['REQ1','REQ2','ASG1','ASG2'])
DQN.plot_data(env.hist[:, 4], "SLA", do_not_save=True)
DQN.plot_data(env.hist[:, 5], "USED", do_not_save=True)







