

"""
There are many ways to implement more complex algorithms for controlling the movements of people and zombies in a simulation. Here are a few examples of more advanced approaches you could consider:

Pathfinding algorithms: You could use algorithms such as A* or Dijkstra's algorithm to allow people and zombies to intelligently navigate the school layout and avoid obstacles. This could be useful for simulating more realistic or strategic movements, such as people trying to find the best route to escape the school or zombies trying to track down nearby survivors.

Flocking algorithms: You could use algorithms such as Boids or Reynolds' steering behaviors to simulate the collective movements of groups of people or zombies. This could be used to model the behavior of crowds or hordes, and could help to create more realistic and dynamic simulations.

Machine learning algorithms: You could use machine learning techniques such as reinforcement learning or decision trees to train people and zombies to make more intelligent decisions about their movements. This could allow the simulation to adapt and improve over time, and could potentially enable the people and zombies to learn from their experiences and make more effective decisions in the future.
"""


import random
from enum import Enum

import numpy as np


# Define the grid-based environment
class CellType(Enum):
    EMPTY = 0
    OBSTACLE = 1
    AGENT = 2
    GOAL = 3


class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.full((height, width), CellType.OBSTACLE)

    def set_obstacle(self, x, y):
        self.grid[y][x] = CellType.OBSTACLE

    def set_goal(self, x, y):
        self.grid[y][x] = CellType.GOAL

    def is_valid_move(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] != CellType.OBSTACLE

    # Depth-first search maze generation algorithm
    def generate_maze(self, start_x, start_y, end_x, end_y):
        def is_valid_cell(x, y):
            return 0 <= x < self.width and 0 <= y < self.height

        def get_neighbors(x, y):
            neighbors = []
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = x + dx, y + dy
                if is_valid_cell(nx, ny):
                    neighbors.append((nx, ny))
            return neighbors

        def remove_wall(x1, y1, x2, y2):
            self.grid[y1][x1] = CellType.EMPTY
            self.grid[y2][x2] = CellType.EMPTY
            self.grid[(y1 + y2) // 2][(x1 + x2) // 2] = CellType.EMPTY

        stack = [(start_x, start_y)]

        while stack:
            x, y = stack[-1]
            self.grid[y][x] = CellType.EMPTY

            neighbors = [
                (nx, ny)
                for nx, ny in get_neighbors(x, y)
                if self.grid[ny][nx] == CellType.OBSTACLE
            ]

            if neighbors:
                nx, ny = neighbors[np.random.randint(len(neighbors))]
                remove_wall(x, y, nx, ny)
                stack.append((nx, ny))
            else:
                stack.pop()

        # Adjusting the handling of even dimensions
        # Ensuring the last row/column are not left entirely empty
        if self.width % 2 == 0:
            for y in range(0, self.height, 2):
                if self.grid[y][self.width - 2] == CellType.EMPTY:
                    self.grid[y][self.width - 1] = CellType.EMPTY
        if self.height % 2 == 0:
            for x in range(0, self.width, 2):
                if self.grid[self.height - 2][x] == CellType.EMPTY:
                    self.grid[self.height - 1][x] = CellType.EMPTY

        self.set_goal(end_x, end_y)

    # Prim's algorithm
    def generate_maze_1(self, start_x, start_y, end_x, end_y):
        def is_valid_cell(x, y):
            return 0 <= x < self.width and 0 <= y < self.height

        def get_neighbors(x, y):
            neighbors = []
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = x + dx, y + dy
                if is_valid_cell(nx, ny):
                    neighbors.append((nx, ny))
            return neighbors

        def remove_wall(x1, y1, x2, y2):
            self.grid[y1][x1] = CellType.EMPTY
            self.grid[y2][x2] = CellType.EMPTY
            self.grid[(y1 + y2) // 2][(x1 + x2) // 2] = CellType.EMPTY

        # Initialize all cells as walls
        self.grid.fill(CellType.OBSTACLE)

        # Start from a random point and make it empty
        self.grid[start_y][start_x] = CellType.EMPTY

        # Add the neighbors of the start cell to the frontier
        frontier = [(start_x, start_y)]

        while frontier:
            current = random.choice(frontier)
            frontier.remove(current)
            x, y = current

            neighbors = [n for n in get_neighbors(x, y) if self.grid[n[1]][n[0]] == CellType.EMPTY]
            if neighbors:
                nx, ny = random.choice(neighbors)
                remove_wall(x, y, nx, ny)

            frontier.extend([n for n in get_neighbors(x, y) if self.grid[n[1]][n[0]] == CellType.OBSTACLE and n not in frontier])

        # Ensure the last row and column are integrated into the maze
        if self.width % 2 == 0:
            for y in range(0, self.height, 2):
                if is_valid_cell(self.width - 3, y):
                    remove_wall(self.width - 3, y, self.width - 2, y)
                    self.grid[y][self.width - 1] = CellType.EMPTY

        if self.height % 2 == 0:
            for x in range(0, self.width, 2):
                if is_valid_cell(x, self.height - 3):
                    remove_wall(x, self.height - 3, x, self.height - 2)
                    self.grid[self.height - 1][x] = CellType.EMPTY

        self.set_goal(end_x, end_y)
        
    # Kruskal's Algorithm
    def generate_maze_2(self, start_x, start_y, end_x, end_y):
        def is_valid_cell(x, y):
            return 0 <= x < self.width and 0 <= y < self.height

        def get_neighbors(x, y):
            neighbors = []
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = x + dx, y + dy
                if is_valid_cell(nx, ny):
                    neighbors.append((nx, ny))
            return neighbors

        def remove_wall(x1, y1, x2, y2):
            self.grid[y1][x1] = CellType.EMPTY
            self.grid[y2][x2] = CellType.EMPTY
            self.grid[(y1 + y2) // 2][(x1 + x2) // 2] = CellType.EMPTY

        # Initialize all cells as walls
        self.grid.fill(CellType.OBSTACLE)

        # Create initial sets for each cell and list of walls
        sets = {}
        walls = []

        # Open each cell and gather walls
        for y in range(0, self.height, 2):
            for x in range(0, self.width, 2):
                self.grid[y][x] = CellType.EMPTY
                sets[(x, y)] = {(x, y)}
                for nx, ny in get_neighbors(x, y):
                    if is_valid_cell(nx, ny) and (nx, ny) not in sets:
                        walls.append(((x, y), (nx, ny)))

        # Shuffle the walls to randomize the process
        random.shuffle(walls)

        # Merge sets and remove walls
        for cell1, cell2 in walls:
            set1 = sets[cell1]
            set2 = sets[cell2]
            if set1 != set2:
                remove_wall(*cell1, *cell2)
                union_set = set1.union(set2)
                for cell in union_set:
                    sets[cell] = union_set

        # Ensure the last row and column are integrated into the maze
        if self.width % 2 == 0:
            for y in range(0, self.height, 2):
                if is_valid_cell(self.width - 3, y):
                    remove_wall(self.width - 3, y, self.width - 2, y)
                    self.grid[y][self.width - 1] = CellType.EMPTY

        if self.height % 2 == 0:
            for x in range(0, self.width, 2):
                if is_valid_cell(x, self.height - 3):
                    remove_wall(x, self.height - 3, x, self.height - 2)
                    self.grid[self.height - 1][x] = CellType.EMPTY

        self.set_goal(end_x, end_y)

    # Eller's Algorithm
    def generate_maze_3(self, start_x, start_y, end_x, end_y):
        def join_sets(sets, x1, y, x2):
            set1 = sets[x1, y]
            set2 = sets[x2, y]
            if set1 != set2:
                for k in sets.keys():
                    if sets[k] == set2:
                        sets[k] = set1
                remove_wall(x1, y, x2, y)

        def remove_wall(x1, y1, x2, y2):
            self.grid[y1][x1] = CellType.EMPTY
            self.grid[y2][x2] = CellType.EMPTY
            self.grid[(y1 + y2) // 2][(x1 + x2) // 2] = CellType.EMPTY

        # Initialize all cells as walls
        self.grid.fill(CellType.OBSTACLE)

        # Initial setup for sets
        sets = {}
        set_counter = 1

        for y in range(0, self.height, 2):
            # Assign sets to each cell in the row
            for x in range(0, self.width, 2):
                self.grid[y][x] = CellType.EMPTY
                if (x, y) not in sets:
                    sets[(x, y)] = set_counter
                    set_counter += 1

            # Randomly join adjacent sets in the same row
            for x in range(0, self.width - 2, 2):
                if random.choice([True, False]):
                    join_sets(sets, x, y, x + 2)

            # Create vertical connections
            if y < self.height - 2:
                next_row_sets = set(sets[x, y] for x in range(0, self.width, 2))
                for set_id in next_row_sets:
                    cells_in_set = [x for x in range(0, self.width, 2) if sets[x, y] == set_id]
                    # Ensure at least one cell in the set connects downward
                    x = random.choice(cells_in_set)
                    sets[(x, y + 2)] = sets[(x, y)]
                    remove_wall(x, y, x, y + 2)
                    cells_in_set.remove(x)
                    # Optionally connect other cells in the set downward
                    for x in cells_in_set:
                        if random.choice([True, False]):
                            sets[(x, y + 2)] = sets[(x, y)]
                            remove_wall(x, y, x, y + 2)

            # Additional handling for the last row
            if y == self.height - 2 or y == self.height - 1:
                # Join all adjacent sets in the last row
                for x in range(0, self.width - 2, 2):
                    if sets[(x, y)] != sets[(x + 2, y)]:
                        join_sets(sets, x, y, x + 2)

        # Additional handling for the last column
        if self.width % 2 == 0:
            last_col = self.width - 1
            for y in range(0, self.height, 2):
                if self.grid[y][last_col - 1] == CellType.EMPTY:
                    self.grid[y][last_col] = CellType.EMPTY
                    if y > 0 and self.grid[y - 1][last_col - 1] == CellType.EMPTY:
                        self.grid[y - 1][last_col] = CellType.EMPTY

        # Merge any separate sets in the last row
        for x in range(0, self.width - 2, 2):
            if sets[(x, self.height - 2)] != sets[(x + 2, self.height - 2)]:
                join_sets(sets, x, self.height - 2, x + 2)

        self.set_goal(end_x, end_y)

# Usage example
grid = Grid(10, 10)
grid.generate_maze(0, 0, 9, 9)

# text based visualization
for y in range(grid.height):
    for x in range(grid.width):
        if grid.grid[y][x] == CellType.OBSTACLE:
            print('#', end='')
        elif grid.grid[y][x] == CellType.EMPTY:
            print('.', end='')
        elif grid.grid[y][x] == CellType.GOAL:
            print('G', end='')
        else:
            print(' ', end='')
    print()



# or wave function collapse
# Greedy Best-First Search
# Bidirectional Search
# Recursive Division
# Growing Tree Algorithm
# https://github.com/avihuxp/WaveFunctionCollapse
# https://www.youtube.com/watch?v=2SuvO4Gi7uY
# https://en.wikipedia.org/wiki/Maze_generation_algorithm
# https://youtu.be/rI_y2GAlQFM
# https://youtu.be/TlLIOgWYVpI
# https://youtu.be/20KHNA9jTsE
# https://youtu.be/TO0Tx3w5abQ
# https://realpython.com/python-maze-solver/

