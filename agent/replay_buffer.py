import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """
    Stores past experiences (state, action, reward, next_state, done)
    and allows random mini-batch sampling.

    Sampling randomly (instead of learning from experiences in the
    order they occurred) breaks the correlation between consecutive
    steps, which is one of the key ideas that makes DQN stable.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a single transition in the buffer.
        """

        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions and return them as
        torch tensors, ready to be used in training.
        """

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.array(states), dtype=torch.float32
        )

        actions = torch.tensor(
            np.array(actions), dtype=torch.int64
        ).unsqueeze(1)

        rewards = torch.tensor(
            np.array(rewards), dtype=torch.float32
        ).unsqueeze(1)

        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float32
        )

        dones = torch.tensor(
            np.array(dones), dtype=torch.float32
        ).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)