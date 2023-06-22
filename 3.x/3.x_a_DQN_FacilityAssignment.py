import sys
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=150)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x) if x%1 else "{0:0.0f}".format(x)})

sys.path.append('../util')
from DQN import DeepQNetwork

sys.path.append('../env')
from a_facility_assignment import FacilityAssignment

discount_rate = 0.9
learning_rate = 0.001 # 0.001
epsilon       = 0.1   # 0.1
n_episodes    = 1000  # 1000
replace       = 1000  # 1000
to_load_model = True
keep_learning = True

env = FacilityAssignment()

DQN = DeepQNetwork(n_features=env.n_features,
                   n_actions=env.n_actions,
                   learning_rate=learning_rate,
                   reward_decay=discount_rate,
                   memory_size=2**16,  # 65536
                   batch_size=128,  # 128
                   replace_target_iter=replace,
                   to_load_model=to_load_model,
                   name_model='a_facility_assignment_DQN')

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return DQN.choose_action(s)
    else:
        return rnd.choice(env.n_actions)

steps_to_start_learning = DQN.batch_size * 10

step      = 0
r_account = np.zeros(4)
steps_per_episode = 0

for episode in range(1, n_episodes+1):

    s = env.reset()
    steps_per_episode = 0

    while True:

        step += 1
        steps_per_episode += 1

        a = epsilon_greed_policy(epsilon, s)

        # simple policy, use the less loaded if in range:
        # in_range = env.state_in_range
        # occupancy = env.state_occupancy
        # order = np.argsort(occupancy)
        # for server in order:
        #     if in_range[server]:
        #         a = server
        #         break

        s_, r, done = env.step(a)

        r_account[r + 2] += 1  # r from [-2,1] -> r+2 from [0,3]

        DQN.memory.store_transition(s, a, r, s_)

        if step > steps_to_start_learning and keep_learning:
            error = DQN.learn()

        s = s_  # move to next state

        if done:
            # r_account[3] = r_account[3] + 30 as the episode concludes
            # when the 30 slots have been occupied.

            # the closer the steps_per_episode to 30 the better, as it
            # says, in all steps, the right server selection has been made.

            if episode % 100 == 0:
                print('episode: ', episode,
                      ', r_account: ', r_account,
                      ', steps_per_episode: ', steps_per_episode)
            break

r_account = r_account / step

print('r_account: ', r_account,
      ' acceptance_ratio: %.3f' % r_account[3])

DQN.save_model()








