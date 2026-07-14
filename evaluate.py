import argparse
import time

import numpy as np

import config
from environment.warehouse import WarehouseEnv
from agent.dqn_agent import DQNAgent


def evaluate(
    model_path,
    episodes=10,
    render=True,
    delay=0.2,
):
    """
    Load a trained agent and run it greedily (no exploration) on
    fresh, randomly generated warehouses.
    """

    env = WarehouseEnv(rows=config.ROWS, cols=config.COLS)
    agent = DQNAgent(state_size=config.ROWS * config.COLS)

    agent.load(model_path)

    # Greedy evaluation: no random exploration. We don't touch
    # config.EPSILON_END here, we just override the agent's live
    # epsilon value for this run.
    agent.epsilon = 0.0

    successes = 0
    episode_rewards = []
    episode_steps = []

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0
        done = False
        steps = 0

        if render:
            print(f"\n=== Episode {episode + 1} / {episodes} ===")
            env.render()

        for step in range(config.MAX_STEPS):

            action = agent.select_action(state)

            next_state, reward, done = env.step(action)

            state = next_state
            total_reward += reward
            steps += 1

            if render:
                env.render()
                time.sleep(delay)

            if done:
                break

        if env.delivered:
            successes += 1

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        if env.delivered:
            status = "DELIVERED"
        elif done:
            status = "STUCK"
        else:
            status = "TIMED OUT"

        print(
            f"Episode {episode + 1:>3}: "
            f"{status:<10} "
            f"Reward: {total_reward:>7.1f} "
            f"Steps: {steps:>3}"
        )

    success_rate = (successes / episodes) * 100

    print("\n=== Evaluation Summary ===")
    print(f"Episodes:        {episodes}")
    print(f"Success Rate:    {success_rate:.1f}%")
    print(f"Average Reward:  {np.mean(episode_rewards):.2f}")
    print(f"Average Steps:   {np.mean(episode_steps):.1f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN warehouse agent."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pth",
        help="Path to the model checkpoint to load.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate.",
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable printing the grid at every step (faster).",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Seconds to pause between rendered steps.",
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        render=not args.no_render,
        delay=args.delay,
    )