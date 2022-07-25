"""
The memory that used for reply,
"""
import numpy as np
import collections


class Memory(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.memory = collections.deque(maxlen=maxlen)
        self.Transition = collections.namedtuple("Transition",
                                                 ("state", "outputs", "action", "reward"))

    def store(self, state, output, action, reward):
        self.memory.append(self.Transition(state, output, action, reward))

    def sample(self, batch_size):
        indexes = np.random.choice(self.maxlen, size=batch_size, replace=False)
        experience_batch = [self.memory[i] for i in indexes]
        IS_weights = [1 for _ in range(batch_size)]
        return indexes, experience_batch, IS_weights

    def full(self):
        if len(self.memory) == self.maxlen:
            return True
        else:
            return False

    def avg_reward(self):
        assert len(self.memory) > 0
        sum_reward = 0
        for i in range(len(self.memory)):
            sum_reward += self.memory[i].reward
        return sum_reward / len(self.memory)
