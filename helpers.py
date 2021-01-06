import random
import numpy as np
import cv2
import gym
from collections import deque


class Experience_Replay(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

    def get_entire_buffer(self):
        return self.buffer


class Prioritized_Experience_Replay():
    def __init__(self, capacity, prob_alpha=0.6, beta=0.4):
        #super(Prioritized_Experience_Replay, self).__init__()
        self.buffer = []
        self.capacity = capacity
        self.prob_alpha = prob_alpha
        self.beta = beta
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state,0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indicies = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indicies]

        N = len(self.buffer)
        weights = (N*probs[indicies]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        state, action, reward, next_state, done = zip(*samples)

        return np.concatenate(state), action, reward, np.concatenate(next_state), done, indicies, weights

    def update_priorities(self, batch_indicies, batch_priorities):
        for idx, prio in zip(batch_indicies, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

    def get_entire_buffer(self):
        return self.buffer




def wrap_env(env):
    return env
