import numpy as np
import numpy.random as rnd

# states:  (A,B,C,D,E,T)
num_states = 6
print("num_states: ", num_states)
# actions: (Left, Right)
num_actions = 2
print("num_actions: ", num_actions)

P = np.array([
    [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
    [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
    [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
    [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
])

R = np.array([
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
])

V = np.full(num_states, 0.5)
V[5] = 0.0

###################################################

discount_rate = 1.00
learning_rate = 0.01
n_steps       = 50000

s = 2  # start in state C

# Temporal Difference algorithm
for step in range(n_steps):
    a = rnd.choice(num_actions)  # random walk policy
    s_ = rnd.choice(num_states, p=P[s, a])  # pick next state using P[s, a]
    error = R[s, a, s_] + discount_rate * V[s_] - V[s]
    V[s] = V[s] + learning_rate * error
    s = s_  # move to next state
    if s == 5: # if s == T, the episode is over
        s = 2

print("\nV(s):", V)
