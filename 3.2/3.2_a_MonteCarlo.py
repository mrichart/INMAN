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
n_episodes    = 30000

seq_states = []
seq_rewards = []

s = 2  # start in state C

# MonteCarlo algorithm
for episode in range(n_episodes):
    a  = rnd.choice(num_actions)  # random walk policy
    s_ = rnd.choice(num_states, p=P[s, a])  # pick next state using P[s, a]
    seq_states.append(s)
    seq_rewards.append(R[s, a, s_])
    s = s_  # move to next state
    if s == 5: # if s == T, the episode is over
        seq_states.reverse()
        seq_rewards.reverse()
        Return = 0
        for state in seq_states:
            Return = seq_rewards.pop(0) + (discount_rate*Return)
            error = Return - V[state]
            V[state] = V[state] + learning_rate * error

        seq_states = []
        seq_rewards = []
        s = 2
        continue

print("\nV(s):", V)
