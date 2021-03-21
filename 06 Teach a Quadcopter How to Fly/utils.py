# https://github.com/rll/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
import numpy as np
from scipy.special import softmax
from collections import deque
from typing import Tuple


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        # transition is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int = 128):
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []

        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))

        return np.array(state), np.array(action), np.array(reward), np.array(
            next_state), np.array(done)
