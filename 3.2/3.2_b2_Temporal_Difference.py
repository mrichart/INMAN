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

V = np.full(num_states, 5.0)

###################################################

discount_rate = 0.95
learning_rate = 0.01
steps = 30000

s = 0  # start in state 0

# Temporal Difference algorithm
for step in range(steps):
    a = rnd.choice(allowed_actions[s])  # random policy
    s_ = rnd.choice(num_states, p=P[s, a])  # pick next state using P[s, a]
    error = R[s, a, s_] + discount_rate * V[s_] - V[s]
    V[s] = V[s] + learning_rate * error
    s = s_  # move to next state

print("\nV(s):", V)
