from collections import deque


class BFSValidator:
    """
    Uses Breadth-First Search (BFS) to determine whether two
    positions in the warehouse are connected.
    """

    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def in_bounds(self, row, col):
        return (
            0 <= row < self.rows and
            0 <= col < self.cols
        )

    def is_walkable(self, row, col):
        return self.grid[row][col] != "#"

    def path_exists(self, start, goal):
        """
        Returns True if a path exists from start to goal.
        """

        queue = deque([start])

        visited = {start}

        directions = [
            (-1, 0),   # Up
            (1, 0),    # Down
            (0, -1),   # Left
            (0, 1)     # Right
        ]

        while queue:

            row, col = queue.popleft()

            if (row, col) == goal:
                return True

            for dr, dc in directions:

                nr = row + dr
                nc = col + dc

                if not self.in_bounds(nr, nc):
                    continue

                if not self.is_walkable(nr, nc):
                    continue

                if (nr, nc) in visited:
                    continue

                visited.add((nr, nc))
                queue.append((nr, nc))

        return False