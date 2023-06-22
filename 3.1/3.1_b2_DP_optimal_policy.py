import numpy as np

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
Q = np.full((num_states, num_actions), -np.inf)

###################################################

discount_rate = 0.95
n_iterations  = 100

# Value Iteration algorithm
for iteration in range(n_iterations):
    V_prev = V.copy()
    for s in range(num_states):
        for a in allowed_actions[s]:
            Q[s, a] = np.sum([P[s, a, s_] *
                              (R[s, a, s_] + discount_rate * V_prev[s_])
                             for s_ in range(num_states)])
        V[s] = np.max(Q[s])

print("\nV(s):", V)

for s in range(num_states):
    for a in allowed_actions[s]:
        Q[s, a] = np.sum([P[s, a, s_] *
                          (R[s, a, s_] + discount_rate * V[s_])
                         for s_ in range(num_states)])

print("\nQ(s,a):")
print(Q)
print("\noptimal policy:", np.argmax(Q, axis=1), ", a to be taken from each state")


