import sys
import gym
import numpy as np

sys.path.append('../util')
from DQN import DeepQNetwork

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

DQN = DeepQNetwork(n_features=env.observation_space.shape[0],
                   n_actions=env.action_space.n,
                   learning_rate=0.001,
                   reward_decay=0.99, # 0.99: 100 steps ahead, 0.995: 200 steps ahead
                   memory_size=2**15, #32768
                   batch_size=2**7, #128
                   replace_target_iter=1000,
                   to_load_model=True,
                   name_model="CartPole-v0")

total_steps = 0
steps_to_start_learning = DQN.batch_size*10

max_steps_in_episode = 500
r_avg_target = 0.90
last_hit = -1
hits = 0
hits_target = 10
exit_for = False

RENDER = True
keep_learning = True

for episode in range(5000):

    if exit_for:
        break

    s = env.reset()
    steps_in_episode = 0
    r_avg_in_episode = 0
    epsilon = init_epsilon = 0.40

    ratio_x = 0.0
    ratio_theta = 0.0

    while True:

        try:
            if RENDER: env.render()
        except KeyboardInterrupt:
            exit_for = True

        total_steps += 1
        steps_in_episode += 1

        if np.random.uniform(0,1) < epsilon:
            a = np.random.randint(0, env.action_space.n)
        else:
            a = DQN.choose_action(s)

        if epsilon > 0.05:
            epsilon -= init_epsilon/50

        s_, r, done, info = env.step(a)

        x, x_dot, theta, theta_dot = s_
        r_x = (env.x_threshold - abs(x)) / env.x_threshold
        r_theta = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians

        r = r_x * r_theta

        if done:
            r = -1.0

        if r_x > 0.8:
            ratio_x += 1.0
        if r_theta > 0.8:
            ratio_theta += 1.0

        r_avg_in_episode = (0.02 * r) + (0.98 * r_avg_in_episode)  # avg of last 50 steps
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
            if episode % 100 == 0 and episode!=0:
                DQN.save_model()
            break

DQN.save_model()
env.close()
del env


