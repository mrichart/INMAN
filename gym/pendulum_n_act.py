import sys
import gym
import numpy as np

sys.path.append('../util')
from DQN import DeepQNetwork

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)

n_actions = 5 # odd number

a_pendulum = np.zeros((n_actions,1))
for i in range(n_actions):
    a_pendulum[i][0] = -env.max_torque + i*(env.max_torque/((n_actions-1)/2))

DQN = DeepQNetwork(n_features=env.observation_space.shape[0],
                   n_actions=n_actions,
                   learning_rate=0.01,
                   reward_decay=0.99,
                   memory_size=2048,
                   batch_size=128,
                   replace_target_iter=3000,
                   to_load_model=True,
                   name_model="Pendulum-v0_5act_double")

total_steps = 0
r_avg_real = 0.0
steps_to_start_learning = DQN.batch_size*10

max_steps_in_episode = 500
r_avg_target = 0.95
last_hit = -1
hits = 0
hits_target = 5
exit_for = False

RENDER = True
keep_learning = True

for episode in range(100):

    if exit_for:
        break

    s = env.reset()
    steps_in_episode = 0
    r_avg_in_episode = 0.0
    epsilon = init_epsilon = 1.0

    while True:

        if RENDER: env.render()

        total_steps += 1
        steps_in_episode += 1

        if np.random.uniform(0,1) < epsilon:
            a = np.random.randint(0, n_actions)
        else:
            a = DQN.choose_action(s)

        if epsilon > 0.01:
            epsilon -= init_epsilon/250

        s_, r, done, info = env.step(a_pendulum[a])

        cos_theta_, sin_theta_, theta_dot_ = s_
        r_theta_dot_ = 1 - (abs(theta_dot_)/env.max_speed)
        r  = cos_theta_ * r_theta_dot_

        r_avg_in_episode = (0.02 * r) + (0.98 * r_avg_in_episode)  # avg of last 50 steps
        r_avg_real = r_avg_real + (1 / total_steps) * (r - r_avg_real)
        DQN.memory.store_transition(s, a, r, s_)
        s = s_

        if total_steps > steps_to_start_learning and keep_learning:
            DQN.learn()

        if steps_in_episode == max_steps_in_episode:
            done = True

        if done:
            print(' Episode: ', episode,
                  ' | epsilon: ', round(epsilon, 2),
                  ' | steps_in_episode: ', steps_in_episode,
                  ' | r_avg_in_episode: ', r_avg_in_episode,
                  ' | r_avg_real: ', r_avg_real,
                  ' | total_steps: ', total_steps
                  )
            if r_avg_in_episode > r_avg_target:
                if last_hit == episode - 1:
                    hits += 1
                else:
                    hits = 1
                last_hit = episode
                if hits == hits_target:
                    exit_for = True
            break

DQN.save_model()


