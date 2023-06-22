import sys
import gym
import numpy as np

sys.path.append('../util')
from DQN import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(1)

DQN = DeepQNetwork(n_features=env.observation_space.shape[0],
                   n_actions=env.action_space.n,
                   learning_rate=0.01,
                   reward_decay=0.99,
                   memory_size=2048,
                   batch_size=128,
                   replace_target_iter=3000,
                   to_load_model=True,
                   name_model="MountainCar-v0")

total_steps = 0
steps_to_start_learning = DQN.batch_size*10

steps_target = 175 # quite limiting, 200 is more conservative
last_hit = -1
hits = 0
hits_target = 10
exit_for = False

RENDER = True
keep_learning = True

for episode in range(1000):

    if exit_for:
        break

    s = env.reset()
    steps_in_episode = 0
    r_avg_in_episode = 0
    epsilon = 0.0
    tries = 0.0

    while True:

        if RENDER: env.render()

        total_steps += 1
        steps_in_episode += 1

        if steps_in_episode % 1000 == 0 :
            epsilon = 1.0
            print("Episode:", episode,
                  " | steps_in_episode:", steps_in_episode)

        if np.random.uniform(0,1) < epsilon:
            a = np.random.randint(0, env.action_space.n)
        else:
            a = DQN.choose_action(s)

        if epsilon > 0.01:
            epsilon -= 0.001 # 0.001 x 1000 = 1.0

        s_, r, done, info = env.step(a)

        position, velocity = s
        position_, velocity_ = s_

        if position_>(-0.5) and position<(-0.5):
            tries +=1

        r = abs(position_ - (-0.5)) * (abs(velocity_) / env.max_speed)

        r_avg_in_episode = (0.02 * r) + (0.98 * r_avg_in_episode)  # avg of last 50 steps
        DQN.memory.store_transition(s, a, r, s_)
        s = s_

        if total_steps > steps_to_start_learning and keep_learning:
            DQN.learn()

        if done:
            print(' Episode: ', episode,
                  ' | epsilon: ', round(epsilon, 2),
                  ' | steps_in_episode: ', steps_in_episode,
                  ' | r_avg_in_episode: ', r_avg_in_episode,
                  ' | total_steps: ', total_steps,
                  ' | tries: ', tries
                  )
            if steps_in_episode < steps_target:
                if last_hit == episode - 1:
                    hits += 1
                else:
                    hits = 1
                last_hit = episode
                if hits == hits_target:
                    exit_for = True
            break

DQN.save_model()

