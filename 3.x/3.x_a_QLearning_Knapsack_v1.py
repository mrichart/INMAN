import sys
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=150)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x) if x%1 else "{0:0.0f}".format(x)})

sys.path.append('../env')
from a_knapsack_v1 import Knapsack

discount_rate = 0.9
learning_rate = 0.001
epsilon       = 0.1      # 0.1
n_episodes    = 1000
load_Q        = False

env = Knapsack()

# Q shape (3,11,2):
Q = np.full((env.n_req_types, env.srv_size+1, env.n_actions), 0.0)

if load_Q:
    Q = np.load("../model/a_knapsack_v1_QLearn_Q.npy")

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return np.argmax(Q[s[0],s[1]])
    else:
        return rnd.choice(env.n_actions)

step = 0

print('req_size: ', env.req_size)
print('req_val:  ', env.req_val)
print('')

for episode in range(n_episodes):

    s = env.reset()
    R = 0
    acts = []

    while True:

        step += 1

        a = epsilon_greed_policy(epsilon, s)  # epsilon greed policy

        if a == 0:
            acts.append(env.req_size[s[0]])
        else:
            acts.append('--')

        s_, r, done = env.step(a)

        R += r

        error = r + discount_rate * np.max(Q[s_[0],s_[1]]) - Q[s[0], s[1], a]
        Q[s[0], s[1], a] += learning_rate * error

        s = s_  # move to next state

        if done:
            if episode % 10 == 0:
                print("content: ", env.state_content,
                      ', R: %.1f' % (R*sum(env.req_val)),
                      ', acts: ', acts)
            break

np.save("../model/a_knapsack_v1_QLearn_Q", Q)






