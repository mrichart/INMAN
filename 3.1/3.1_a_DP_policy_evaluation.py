import numpy as np

# states:  (s0, s1, s2)
print("num_states: 3")
# actions: (a0, a1)
print("num_actions: 2")

# Transition probabilities from s, taken a, visited s_: P(s, a, s_)
# shape (3,2,3)
P = np.array([
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ])

print("\nshape of P:", P.shape)

# Expected reward of a: R(s, a, s_)
# shape (3,2,3)
R = np.array([
    [[0.0, 5.0, 0.0], [0.0, 0.0, 10.0]],
    [[5.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
    [[5.0, 0.0, 0.0], [0.0, 5.0,  0.0]], ])

print("shape of R:", R.shape)

# policy pi(s,a)
# shape (3,2)
pi = np.array([[1/3, 2/3],
               [1/3, 2/3],
               [1/2, 1/2], ])

print("shape of pi:", pi.shape)

# shape of V: (3,)
# shape of Q: (3,2)

V_init = np.full(pi.shape[0], 5.0)
print("shape of V:", V_init.shape)
Q = np.full(pi.shape, 0.0)
print("shape of Q:", Q.shape)

discount_rate = 0.5
n_iterations = 100

num_states  = pi.shape[0]
num_actions = pi.shape[1]

# iterative policy evaluation algorithm
def iterative_policy_evaluation(V_init, pi):
    V_new = V_init.copy()
    for iteration in range(n_iterations):
        V_prev = V_new.copy()
        for s in range(num_states):
            V_new[s] = np.sum(pi[s, a] *
                          np.sum([P[s, a, s_] *
                                  (R[s, a, s_] + discount_rate * V_prev[s_])
                                  for s_ in range(num_states)])
                          for a in range(num_actions))
    return V_new

def value_of_Q(V):
    Q_new = np.full(pi.shape, 0.0)
    for s in range(num_states):
        for a in range(num_actions):
            Q_new[s, a] = np.sum([P[s, a, s_] *
                              (R[s, a, s_] + discount_rate * V[s_])
                              for s_ in range(num_states)])
    return Q_new

###################################################

V_new = iterative_policy_evaluation(V_init, pi)
print("\nV(s):", V_new)

Q_new = value_of_Q(V_new)
print("\nQ(s,a):")
print(Q_new)

###################################################

print("\nimproved policy:", np.argmax(Q_new, axis=1), ", a to be taken from each state.")

# Policy improvement algorithm
while True:

    pi_new = np.zeros((num_states, num_actions))
    for state, action in enumerate(np.argmax(Q_new, axis=1)):
        pi_new[state, action] = 1.0
    print('new_pi:')
    print(pi_new)

    V_old = V_new
    V_new = iterative_policy_evaluation(V_old, pi_new)
    print("\nV(s):", V_new)

    Q_old = Q_new
    Q_new = value_of_Q(V_new)
    print("\nQ(s,a):")
    print(Q_new)

    print("\nimproved policy:", np.argmax(Q_new, axis=1), ", a to be taken from each state")

    if (np.argmax(Q_old, axis=1) == np.argmax(Q_new, axis=1)).all():
        print("policy repeated, last policy is already optimal.")
        break





