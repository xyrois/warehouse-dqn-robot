import random

import numpy as np

from environment.generator import WarehouseGenerator


class WarehouseEnv:
    """
    Reinforcement Learning environment for the warehouse robot.
    """

    MOVE_REWARD = -1
    WALL_REWARD = -5
    PACKAGE_REWARD = 50
    DELIVERY_REWARD = 200

    # Reward shaping: +1 if the robot's move decreased its distance
    # to the current goal (package, or delivery once carrying),
    # -1 if it increased. This gives a training signal on every
    # single step instead of only at pickup/delivery, which are
    # otherwise many steps apart.
    DISTANCE_SHAPING_REWARD = 1

    # If the agent bumps a wall/boundary this many times in a row,
    # end the episode early instead of letting it burn the rest of
    # MAX_STEPS accumulating WALL_REWARD.
    MAX_CONSECUTIVE_INVALID = 10
    STUCK_REWARD = -50

    def __init__(
        self,
        rows=10,
        cols=10,
        obstacle_probability=0.20,
        fixed_maps=None,
    ):
        """
        fixed_maps: optional list of pre-generated warehouse dicts
        (as returned by WarehouseGenerator.generate()). If provided,
        reset() will pick a random map from this pool instead of
        generating a brand-new one. This supports curriculum
        learning: train on 1 map, then a handful, before moving to
        fully random maps (fixed_maps=None).
        """

        self.rows = rows
        self.cols = cols

        self.generator = WarehouseGenerator(
            rows=rows,
            cols=cols,
            obstacle_probability=obstacle_probability,
        )

        self.fixed_maps = fixed_maps

        self.reset()

    def reset(self):
        """
        Load a new warehouse: either a fresh random one, or one
        drawn from self.fixed_maps if a fixed pool was provided.
        """

        if self.fixed_maps:
            warehouse = random.choice(self.fixed_maps)
        else:
            warehouse = self.generator.generate()

        # Copy the grid so nothing in this episode can accidentally
        # mutate a shared map from the fixed pool.
        self.grid = [row[:] for row in warehouse["grid"]]

        self.robot = warehouse["robot"]
        self.package = warehouse["package"]
        self.delivery = warehouse["delivery"]

        self.carrying = False
        self.delivered = False

        self.consecutive_invalid = 0

        return self.get_state()

    def get_state(self):
        """
        Return the current warehouse as a flattened numeric state.

        Encoding:
            0 = Empty
            1 = Wall
            2 = Robot
            3 = Package
            4 = Delivery
        """

        state = np.zeros(
            (self.rows, self.cols),
            dtype=np.float32
        )

        # Walls
        for r in range(self.rows):
            for c in range(self.cols):

                if self.grid[r][c] == "#":
                    state[r][c] = 1

        # Delivery
        dr, dc = self.delivery
        state[dr][dc] = 4

        # Package
        if not self.carrying:
            pr, pc = self.package
            state[pr][pc] = 3

        # Robot
        rr, rc = self.robot
        state[rr][rc] = 2

        # Flatten into a 100-element vector
        return state.flatten()

    def current_goal(self):
        """
        The position the robot is currently trying to reach:
        the package if it isn't carrying it yet, otherwise the
        delivery point.
        """

        return self.delivery if self.carrying else self.package

    @staticmethod
    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, action):
        """
        Actions:
        0 = Up
        1 = Down
        2 = Left
        3 = Right
        """

        row, col = self.robot

        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1
        else:
            raise ValueError(f"Invalid action: {action}")

        # Outside map
        out_of_bounds = (
            row < 0
            or row >= self.rows
            or col < 0
            or col >= self.cols
        )

        # Hit wall (only check if still in bounds)
        hit_wall = (
            not out_of_bounds
            and self.grid[row][col] == "#"
        )

        if out_of_bounds or hit_wall:

            self.consecutive_invalid += 1

            reward = self.WALL_REWARD
            done = False

            if self.consecutive_invalid >= self.MAX_CONSECUTIVE_INVALID:
                reward += self.STUCK_REWARD
                done = True
                self.delivered = False

            return self.get_state(), reward, done

        # A valid move resets the stuck counter.
        self.consecutive_invalid = 0

        # Reward shaping: compare distance-to-goal before and after
        # the move (goal = package if not carrying, else delivery).
        goal = self.current_goal()
        old_distance = self.manhattan_distance(self.robot, goal)

        # Move robot
        self.robot = (row, col)

        new_distance = self.manhattan_distance(self.robot, goal)

        reward = self.MOVE_REWARD

        if new_distance < old_distance:
            reward += self.DISTANCE_SHAPING_REWARD
        elif new_distance > old_distance:
            reward -= self.DISTANCE_SHAPING_REWARD

        done = False

        # Pick up package
        if (
            not self.carrying
            and self.robot == self.package
        ):
            self.carrying = True
            reward += self.PACKAGE_REWARD

        # Deliver package
        if (
            self.carrying
            and self.robot == self.delivery
        ):
            reward += self.DELIVERY_REWARD
            done = True
            self.delivered = True

        return self.get_state(), reward, done

    def render(self):
        """
        Print the current warehouse.
        """

        print()

        for r in range(self.rows):

            row_output = []

            for c in range(self.cols):

                if (r, c) == self.robot:
                    row_output.append("R")

                elif (
                    not self.carrying
                    and (r, c) == self.package
                ):
                    row_output.append("P")

                elif (r, c) == self.delivery:
                    row_output.append("D")

                else:
                    row_output.append(self.grid[r][c])

            print(" ".join(row_output))

        print()

    def sample_action(self):
        """
        Random action for testing.
        """

        return np.random.randint(4)