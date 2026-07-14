import os

import numpy as np
import matplotlib.pyplot as plt


def plot_history(
    reward_history, loss_history, epsilon_history, success_history=None
):
    """
    Save plots that tell the story of training:
        1. Episode reward (raw + smoothed)
        2. Average loss per episode
        3. Epsilon decay
        4. Success rate (rolling 100-episode window), if provided
    """

    os.makedirs("plots", exist_ok=True)

    # Reward plot
    plt.figure()

    plt.plot(reward_history, alpha=0.3, label="Raw reward")

    if len(reward_history) >= 100:
        smoothed = [
            np.mean(reward_history[max(0, i - 100):i + 1])
            for i in range(len(reward_history))
        ]
        plt.plot(smoothed, label="Average reward (100 ep)")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward")
    plt.legend()
    plt.savefig("plots/reward.png")
    plt.close()

    # ----------------------------
    # Loss plot
    # ----------------------------

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Training Loss")
    plt.savefig("plots/loss.png")
    plt.close()

    # ----------------------------
    # Epsilon plot
    # ----------------------------

    plt.figure()
    plt.plot(epsilon_history)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.savefig("plots/epsilon.png")
    plt.close()

    # ----------------------------
    # Success rate plot
    # ----------------------------

    if success_history is not None and len(success_history) >= 100:

        success_rate = [
            np.mean(success_history[max(0, i - 100):i + 1]) * 100
            for i in range(len(success_history))
        ]

        plt.figure()
        plt.plot(success_rate)
        plt.xlabel("Episode")
        plt.ylabel("Success Rate % (last 100 episodes)")
        plt.title("Delivery Success Rate")
        plt.savefig("plots/success_rate.png")
        plt.close()


def load_history(plots_dir="plots"):
    """
    Load previously saved training history arrays from disk, so
    plots can be regenerated without re-running training.

    Returns a tuple of lists: (reward, loss, epsilon, success).
    Any array not found on disk comes back as None.
    """

    def _load(name):
        path = os.path.join(plots_dir, f"{name}.npy")

        if not os.path.exists(path):
            return None

        return np.load(path).tolist()

    reward_history = _load("reward_history")
    loss_history = _load("loss_history")
    epsilon_history = _load("epsilon_history")
    success_history = _load("success_history")

    return reward_history, loss_history, epsilon_history, success_history


if __name__ == "__main__":

    # Running this file directly regenerates plots/*.png from the
    # history arrays saved by the last training run, without
    # needing to retrain anything.

    (
        reward_history,
        loss_history,
        epsilon_history,
        success_history,
    ) = load_history()

    if reward_history is None:
        print(
            "No saved training history found in plots/. "
            "Run train.py first."
        )
    else:
        plot_history(
            reward_history, loss_history, epsilon_history, success_history
        )
        print("Plots regenerated in plots/")