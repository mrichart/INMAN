import numpy as np
import numpy.random as rnd

class FacilityAssign:

    def __init__(self):

        self.n_servers       = 3
        self.n_resources     = 10.0
        self.max_distance    = 1.0
        self.prob_finish_use = 0.0333

        self.server_coord    = np.array(((0, 0), (0.5, 0), (0, 0.5)))
        self.state_distances = np.zeros(self.n_servers)
        self.state_occupancy = np.zeros(self.n_servers, dtype=int)

        self.n_actions  = self.n_servers
        self.n_features = len(self.observation())

    def observation(self):
        return np.concatenate((self.state_distances, self.n_resources - self.state_occupancy))

    def reset(self):
        self.state_distances = np.zeros(self.n_servers)
        self.state_occupancy = np.zeros(self.n_servers, dtype=int)
        self.step_load()
        return self.observation()

    def step(self, a):

        done = False

        reward = 0
        if self.state_distances[a] <= self.max_distance and self.state_occupancy[a] < self.n_resources:
            self.state_occupancy[a] += 1
            reward = 1
        elif self.state_distances[a] > self.max_distance:
            reward = -2 # out of range of the server
        else:
            for i in range(self.n_servers):
                if self.state_distances[i] <= self.max_distance and self.state_occupancy[i] < self.n_resources:
                    reward = -1 # it could be assigned to this server
                    break

        self.step_processing()
        self.step_load()
        s_ = self.observation()
        return s_, reward, done

    def render(self):
        print("state: ", self.observation())

    def step_processing(self):
        for server in range(self.n_servers):
            num_free = 0
            for resource in range(self.state_occupancy[server]):
                if rnd.random() < self.prob_finish_use:
                    num_free += 1
            self.state_occupancy[server] -= num_free

    def step_load(self):

        server = rnd.randint(self.n_servers)
        dist = rnd.random()
        angle = rnd.random()

        X = self.server_coord[server][0] + dist * np.cos(2 * np.pi * angle)
        Y = self.server_coord[server][1] + dist * np.sin(2 * np.pi * angle)

        for server in range(self.n_servers):
            self.state_distances[server] = np.sqrt(np.square(self.server_coord[server][0] - X)
                                                   + np.square(self.server_coord[server][1] - Y))

if __name__ == "__main__":

    RENDER = True
    env = FacilityAssign()
    s_0 = env.reset()
    R   = 0
    act = []

    for step in range(100):

        a = np.random.randint(0, env.n_actions)
        act.append(a)
        s_, r, done = env.step(a)
        R += r
        if done:
            if RENDER:
                print(env.state_occupancy, " R: ", R, "\tact: ", act)
            s_ = env.reset()
            R  = 0
            act = []


