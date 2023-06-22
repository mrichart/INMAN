import numpy as np

class ResetEpisodes:

    def __init__(self, max_num_episodes):

        self.num_blocks     = min(4, max_num_episodes)
        self.block_size     = int(max_num_episodes / self.num_blocks)
        self.reset_episodes = np.zeros(self.num_blocks, dtype=int)

    def add(self, episode):
        position = int(episode / self.block_size)
        if position == self.num_blocks:
            position -= 1
        self.reset_episodes[position] += 1

    def show(self):
        return self.reset_episodes
