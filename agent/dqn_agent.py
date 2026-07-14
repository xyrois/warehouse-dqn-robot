import random

import torch
import torch.nn as nn
import torch.optim as optim

from agent.dqn import DQN
from agent.replay_buffer import ReplayBuffer

import config


class DQNAgent:
    """
    Encapsulates the full DQN algorithm:

        - Policy network (being trained)
        - Target network (stable copy, used for targets)
        - Optimizer
        - Replay buffer
        - Epsilon-greedy action selection
        - Learning step (sampling + backprop)
        - Target network updates
        - Save / load checkpoints
    """

    def __init__(
        self,
        state_size=100,
        action_size=4,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Networks
        self.policy_net = DQN(
            input_size=state_size,
            hidden_size=config.HIDDEN_SIZE,
            output_size=action_size,
        ).to(self.device)

        self.target_net = DQN(
            input_size=state_size,
            hidden_size=config.HIDDEN_SIZE,
            output_size=action_size,
        ).to(self.device)

        # Target net starts as an exact copy of the policy net,
        # and is never trained directly (only copied into).
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
        self.target_net.eval()

        # Optimizer + loss
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.LEARNING_RATE,
        )

        # Huber loss (smooth L1) behaves like MSE for small errors
        # but like MAE for large ones, so a single wildly-off
        # Q-value prediction doesn't produce an exploding gradient.
        # This matters a lot here because episodes with repeated
        # wall bumps (-10 each) can produce large-magnitude targets.
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)

        # Exploration
        self.epsilon = config.EPSILON_START

    def select_action(self, state):
        """
        Epsilon-greedy action selection.

        With probability epsilon, pick a random action (explore).
        Otherwise, pick the action with the highest predicted
        Q-value (exploit).
        """

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        """

        self.replay_buffer.push(
            state, action, reward, next_state, done
        )

    def learn(self):
        """
        Sample a mini-batch from the replay buffer and perform
        one gradient descent step on the policy network.

        Does nothing if the buffer doesn't have enough experiences
        yet to fill a full batch.
        """

        if len(self.replay_buffer) < config.BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(config.BATCH_SIZE)
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q-values for the actions actually taken
        q_values = self.policy_net(states).gather(1, actions)

        # Best next-state Q-value, according to the target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(
                dim=1, keepdim=True
            )[0]

        # Bellman target. (1 - dones) zeroes out the future term
        # for transitions where the episode ended.
        targets = rewards + config.GAMMA * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients so a single bad/extreme batch can't throw
        # the network's weights into a bad region in one step.
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=10.0
        )

        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """
        Copy the policy network's weights into the target network.
        Called periodically (every config.TARGET_UPDATE episodes).
        """

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def decay_epsilon(self):
        """
        Decay epsilon after each episode, down to a minimum floor.
        """

        self.epsilon = max(
            config.EPSILON_END,
            self.epsilon * config.EPSILON_DECAY,
        )

    def save(self, path=None):
        """
        Save the policy network, optimizer state, and epsilon to
        disk. Saving the optimizer and epsilon (not just the model
        weights) means training can be resumed later without losing
        momentum/adaptive learning-rate state or restarting
        exploration from scratch.
        """

        path = path or config.MODEL_PATH

        torch.save(
            {
                "model": self.policy_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path=None):
        """
        Load a checkpoint into the policy and target networks,
        restoring the optimizer state and epsilon if present.
        """

        path = path or config.MODEL_PATH

        checkpoint = torch.load(path, map_location=self.device)

        # Support old-style checkpoints that were just a raw
        # state_dict (no "model" key).
        if "model" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["model"])
            self.target_net.load_state_dict(checkpoint["model"])

            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            if "epsilon" in checkpoint:
                self.epsilon = checkpoint["epsilon"]
        else:
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)