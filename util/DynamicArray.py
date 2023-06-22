import numpy as np

class DynamicArray:

    def __init__(self):
        self.data = np.zeros((100,2))
        self.capacity = len(self.data)
        self.pointer = 0

    def append(self, value):
        if self.pointer == self.capacity:
            self.capacity *= 4
            new_data = np.zeros((self.capacity,2))
            new_data[:self.pointer] = self.data
            self.data = new_data

        self.data[self.pointer] = value
        self.pointer += 1

    def size(self):
        return self.pointer

    def all_elements(self):
        all = self.data[:self.pointer]
        return all[:,0] / all[:,1]

    def all_elements_in_window(self, wnd_size):
        all = self.data[self.pointer-wnd_size:self.pointer]
        return all[:,0] / all[:,1]

    def avg_all(self):
        all = self.data[:self.pointer]
        return np.sum(all[:,0]) / np.sum(all[:,1])

    def avg_tillN(self, N):
        window = self.data[:N]
        return np.sum(window[:, 0]) / np.sum(window[:, 1])

    def avg_tillN_in_window(self, N, wnd_size):
        window = self.data[N - wnd_size:N]
        return np.sum(window[:, 0]) / np.sum(window[:, 1])

    def cumulative_avg_all(self):
        series = np.zeros(self.size())
        for episode in range(self.size()):
            series[episode]=self.avg_tillN(self.size() - episode)
        return series[::-1]

    def cumulative_avg_in_window(self, window):
        series = np.zeros(window)
        for episode in range(window):
            series[episode]=self.avg_tillN_in_window(self.size() - episode, window - episode)
        return series[::-1]

    def cumulative_avg_all_lastN(self, lastNentries):
        series = np.zeros(lastNentries)
        for episode in range(lastNentries):
            series[episode]=self.avg_tillN(self.size() - episode)
        return series[::-1]

    def cumulative_avg_sliding_window_lastN(self, lastNentries, window):
        series = np.zeros(lastNentries)
        for episode in range(lastNentries):
            series[episode]=self.avg_tillN_in_window(self.size() - episode, window)
        return series[::-1]

    def stop_criterium_all(self, lastNentries, threshold):
        if self.size() >= lastNentries:
            series_lastN = self.cumulative_avg_all_lastN(lastNentries)
            #print("std of all:", series_lastN.std())
            if series_lastN.std() < (threshold * series_lastN.mean()):
                return True
        return False

    def stop_criterium_window(self, lastNentries, wnd, threshold):
        if self.size() >= lastNentries+wnd:
            series_lastN = self.cumulative_avg_sliding_window_lastN(lastNentries, wnd)
            #print("std of wnd:", series_lastN.std())
            if series_lastN.std() < (threshold * series_lastN.mean()):
                return True
        return False