import numpy as np
import threading
from DynamicArray import DynamicArray

class GlobalEpisode:

    def __init__(self, rDC):
        self.rDC = rDC

        self.episode = 0

        self.r_episode_series = DynamicArray()
        self.acts_total       = np.zeros(rDC.env.n_actions, dtype=int)

        self.lock = threading.Lock()

    def done(self, r_sum_in_episode, steps_in_episode, acts_in_episode):
        self.lock.acquire()

        self.r_episode_series.append([r_sum_in_episode, steps_in_episode])
        self.acts_total += acts_in_episode

        if self.episode % self.rDC.save_model_every == 0 and self.episode != 0:
            self.rDC.globalMLT.save_model(self.saver)

        tmp_episode = self.episode
        if self.r_episode_series.size() < 10: tmp_r_avg = self.r_episode_series.avg_all()
        else: tmp_r_avg = self.r_episode_series.avg_tillN_in_window(self.r_episode_series.size(), 10)
        self.episode += 1

        self.lock.release()
        return tmp_episode, tmp_r_avg

    def setSaver(self, saver):
        self.saver = saver
