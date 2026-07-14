import os

import numpy as np

import config
from environment.warehouse import WarehouseEnv
from agent.dqn_agent import DQNAgent
from utils.visualization import plot_history


def train():

    os.makedirs("models", exist_ok=True)

    fixed_maps = None

    if config.NUM_FIXED_MAPS:
        # Pre-generate a small pool of maps up front and reuse them
        # every episode, instead of generating a brand-new random
        # map each time. See config.NUM_FIXED_MAPS for why.
        from environment.generator import WarehouseGenerator

        map_generator = WarehouseGenerator(
            rows=config.ROWS, cols=config.COLS
        )

        fixed_maps = [
            map_generator.generate()
            for _ in range(config.NUM_FIXED_MAPS)
        ]

        print(
            f"Training on a fixed pool of {config.NUM_FIXED_MAPS} "
            f"map(s) (curriculum mode).\n"
        )

    env = WarehouseEnv(
        rows=config.ROWS, cols=config.COLS, fixed_maps=fixed_maps
    )
    agent = DQNAgent(state_size=config.ROWS * config.COLS)

    reward_history = []
    loss_history = []
    epsilon_history = []
    success_history = []
    pickup_history = []

    best_reward = float("-inf")
    training_steps = 0

    for episode in range(config.EPISODES):

        state = env.reset()

        total_reward = 0
        episode_losses = []
        episode_delivered = False

        for step in range(config.MAX_STEPS):

            action = agent.select_action(state)

            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            loss = agent.learn()

            if loss is not None:
                episode_losses.append(loss)
                training_steps += 1

                # Update the target network every N *training
                # steps* rather than every N episodes. Episodes can
                # vary wildly in length (20 steps vs 500 steps), so
                # a step-based schedule keeps updates consistent.
                if training_steps % config.TARGET_UPDATE == 0:
                    agent.update_target()

            state = next_state
            total_reward += reward

            if done:
                episode_delivered = env.delivered
                break

        agent.decay_epsilon()

        # Track history
        reward_history.append(total_reward)

        average_loss = (
            np.mean(episode_losses) if episode_losses else 0.0
        )
        loss_history.append(average_loss)

        epsilon_history.append(agent.epsilon)

        success_history.append(1 if episode_delivered else 0)

        # env.carrying stays True once the package is picked up
        # (it's only reset in reset()), so checking it right after
        # the episode ends tells us whether pickup happened at all
        # -- regardless of whether delivery was also completed.
        pickup_history.append(1 if env.carrying else 0)

        window = min(100, len(reward_history))

        average_reward = np.mean(reward_history[-window:])
        success_rate = np.mean(success_history[-window:]) * 100
        pickup_rate = np.mean(pickup_history[-window:]) * 100

        # Checkpointing
        if average_reward > best_reward:
            best_reward = average_reward
            agent.save("models/best_model.pth")

        if (episode + 1) % 500 == 0:
            agent.save("models/latest_model.pth")

        # Logging
        if (episode + 1) % 100 == 0:
            print(
                f"Episode:            {episode + 1} / {config.EPISODES}\n"
                f"Average Reward:     {average_reward:.2f}\n"
                f"Best Reward:        {best_reward:.2f}\n"
                f"Average Loss:       {average_loss:.4f}\n"
                f"Epsilon:            {agent.epsilon:.3f}\n"
                f"Pickup Rate (100):  {pickup_rate:.1f}%\n"
                f"Success Rate (100): {success_rate:.1f}%\n"
                f"Replay Size:        {len(agent.replay_buffer)}\n"
            )

    agent.save(config.MODEL_PATH)

    # Save raw histories so plots can be regenerated later
    # without retraining.
    os.makedirs("plots", exist_ok=True)

    np.save("plots/reward_history.npy", np.array(reward_history))
    np.save("plots/loss_history.npy", np.array(loss_history))
    np.save("plots/epsilon_history.npy", np.array(epsilon_history))
    np.save("plots/success_history.npy", np.array(success_history))
    np.save("plots/pickup_history.npy", np.array(pickup_history))

    plot_history(
        reward_history, loss_history, epsilon_history, success_history
    )


if __name__ == "__main__":
    train()