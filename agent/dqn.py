import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network.

    Input:
        100-dimensional flattened warehouse state

    Output:
        Q-values for the four actions:
            0 = Up
            1 = Down
            2 = Left
            3 = Right
    """

    def __init__(
        self,
        input_size=100,
        hidden_size=128,
        output_size=4,
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, hidden_size),

            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),

            nn.ReLU(),

            nn.Linear(hidden_size, output_size)

        )

    def forward(self, x):
        return self.network(x)