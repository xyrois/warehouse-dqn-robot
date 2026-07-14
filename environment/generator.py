import random

from environment.bfs import BFSValidator


class WarehouseGenerator:
    def __init__(self, rows, cols, obstacle_probability=0.20):
        self.rows = rows
        self.cols = cols
        self.obstacle_probability = obstacle_probability

    def random_empty_position(self, grid):
        """
        Return a random empty position on the grid.
        """
        while True:
            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.cols - 1)

            if grid[row][col] == ".":
                return (row, col)

    def generate(self):
        """
        Generate a random warehouse that is guaranteed to be solvable.

        The grid only contains:
            "." = empty
            "#" = wall

        Robot, package, and delivery positions are stored separately.
        """

        while True:

            # Generate obstacle map
            grid = []

            for _ in range(self.rows):

                row = []

                for _ in range(self.cols):

                    if random.random() < self.obstacle_probability:
                        row.append("#")
                    else:
                        row.append(".")

                grid.append(row)

            # Random positions
            robot = self.random_empty_position(grid)

            while True:
                package = self.random_empty_position(grid)

                if package != robot:
                    break

            while True:
                delivery = self.random_empty_position(grid)

                if (
                    delivery != robot
                    and delivery != package
                ):
                    break

            # Validate map
            validator = BFSValidator(grid)

            if (
                validator.path_exists(robot, package)
                and
                validator.path_exists(package, delivery)
            ):
                return {
                    "grid": grid,
                    "robot": robot,
                    "package": package,
                    "delivery": delivery,
                }