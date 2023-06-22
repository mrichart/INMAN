import numpy as np
import numpy.random as rnd

# represents impossible actions
nan = np.nan

# states:  (s0, s1, s2)
num_states = 3
print("num_states: ", num_states)
# actions: (a0, a1, a2)
num_actions = 3
print("num_actions: ", num_actions)

P = np.array([
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
    [[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]], ])

R = np.array([
    [[10., 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[10., 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -50.]],
    [[nan, nan, nan], [40., 0.0, 0.0], [nan, nan, nan]], ])

# allowed actions from each state:
allowed_actions = [[0, 1, 2], [0, 2], [1]]

# -inf for impossible actions
Q = np.full((num_states, num_actions), -np.inf)

# Initial value = 0.0, for all possible actions
# func enumerate returns an enumerate object: [(0, [0, 1, 2]), (1, [0, 2]), (2, [1])]
#print(list(enumerate(allowed_actions)))
for state, actions in enumerate(allowed_actions):
    Q[state, actions] = 0.0

def epsilon_greed_policy(epsilon, s):
    if rnd.random() > epsilon:
        return np.argmax(Q[s])
    else:
        return rnd.choice(allowed_actions[s])

###################################################

discount_rate = 0.95
learning_rate = 0.01
epsilon = 0.3
steps = 50000

s = 0  # start in state 0
a = epsilon_greed_policy(epsilon, s)  # epsilon greed policy

# SARSA algorithm
for step in range(steps):
    s_ = rnd.choice(num_states, p=P[s, a])  # pick next state using P[s, a]
    a_ = epsilon_greed_policy(epsilon, s_)   # and the a from s_
    error = R[s, a, s_] + discount_rate * Q[s_,a_] - Q[s,a]
    Q[s, a] = Q[s, a] + learning_rate * error
    s = s_  # move to next state
    a = a_  # having taken next a

print("\nQ(s,a): ")
print(Q)
print("\noptimal policy:", np.argmax(Q, axis=1), ", a to be taken from each state")
