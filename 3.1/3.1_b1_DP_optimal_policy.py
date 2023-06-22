import numpy as np

# states:  (s0, s1, s2)
num_states = 3
print("num_states: ", num_states)
# actions: (a0, a1)
num_actions = 2
print("num_actions: ", num_actions)

# Transition probabilities from s, taken a, visited s_: P(s, a, s_)
# shape (3,2,3)
P = np.array([
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ])

print("\nshape of P:", P.shape)

# Expected reward of (s, a, s_)
# shape (3,2,3)
R = np.array([
    [[0.0, 5.0, 0.0], [0.0, 0.0, 10.0]],
    [[5.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
    [[5.0, 0.0, 0.0], [0.0, 5.0,  0.0]], ])

print("shape of R:", R.shape)

# shape of V: (3,)
# shape of Q: (3,2)

V = np.full(num_states, 5.0)
print("shape of V:", V.shape)
Q = np.full((num_states, num_actions), 0.0)
print("shape of Q:", Q.shape)

###################################################

discount_rate = 0.5
n_iterations = 100

# Value Iteration algorithm
for iteration in range(n_iterations):
    V_prev = V.copy()
    for s in range(num_states):
        for a in range(num_actions):
            Q[s, a] = np.sum([P[s, a, s_] *
                              (R[s, a, s_] + discount_rate * V_prev[s_])
                             for s_ in range(num_states)])
        V[s] = np.max(Q[s])

print("\nV(s):", V)

for s in range(num_states):
    for a in range(num_actions):
        Q[s, a] = np.sum([P[s, a, s_] *
                          (R[s, a, s_] + discount_rate * V[s_])
                         for s_ in range(num_states)])

print("\nQ(s,a):")
print(Q)
print("\noptimal policy:", np.argmax(Q, axis=1), ", a to be taken from each state")
