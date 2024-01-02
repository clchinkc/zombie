

"""
There are many ways to implement more complex algorithms for controlling the movements of people and zombies in a simulation. Here are a few examples of more advanced approaches you could consider:

Pathfinding algorithms: You could use algorithms such as A* or Dijkstra's algorithm to allow people and zombies to intelligently navigate the school layout and avoid obstacles. This could be useful for simulating more realistic or strategic movements, such as people trying to find the best route to escape the school or zombies trying to track down nearby survivors.

Flocking algorithms: You could use algorithms such as Boids or Reynolds' steering behaviors to simulate the collective movements of groups of people or zombies. This could be used to model the behavior of crowds or hordes, and could help to create more realistic and dynamic simulations.

Machine learning algorithms: You could use machine learning techniques such as reinforcement learning or decision trees to train people and zombies to make more intelligent decisions about their movements. This could allow the simulation to adapt and improve over time, and could potentially enable the people and zombies to learn from their experiences and make more effective decisions in the future.
"""


import heapq
import math
import random
from collections import defaultdict, deque
from enum import Enum
from queue import PriorityQueue

import numpy as np
import pygame


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


# text based visualization
def print_grid(grid):
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

# Usage example
# grid = Grid(10, 10)
# grid.generate_maze_3(0, 0, 9, 9)
# print_grid(grid)



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




# Create agent classes
class Agent:
    def __init__(self, x, y, algorithm):
        self.x = x
        self.y = y
        self.algorithm = algorithm

    def move(self, grid):
        self.x, self.y = self.algorithm.move(self.x, self.y, grid)

class RandomWalk:
    def __init__(self):
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def move(self, x, y, grid):
        random.shuffle(self.directions)
        for dx, dy in self.directions:
            new_x, new_y = x + dx, y + dy
            if grid.is_valid_move(new_x, new_y):
                return new_x, new_y
        return x, y


class AStar:
    def __init__(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.heuristic_cache = {}

    def heuristic(self, x, y):
        if (x, y) not in self.heuristic_cache:
            self.heuristic_cache[(x, y)] = abs(x - self.goal_x) + abs(y - self.goal_y)
        return self.heuristic_cache[(x, y)]

    def find_neighbors(self, grid, pos):
        x, y = pos
        neighbors = [(x + dy, y + dx) for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
        valid_neighbors = [neighbor for neighbor in neighbors if grid.is_valid_move(*neighbor)]
        return valid_neighbors

    def find_path(self, start_pos, grid):
        end_pos = (self.goal_x, self.goal_y)

        # Use a priority queue to store nodes based on their f-score (total cost).
        open_list = PriorityQueue()
        open_list.put((0, self.heuristic(*start_pos), start_pos))

        # Use a set to check for membership in the open list efficiently.
        open_set = set([start_pos])

        visited = set()

        g_scores = {start_pos: 0}
        f_scores = {start_pos: self.heuristic(*start_pos)}

        paths = {start_pos: [start_pos]}  # Use a dictionary to store paths

        while not open_list.empty():
            _, _, current = open_list.get()
            
            # If the current node is not in the open set, skip processing it.
            if current not in open_set:
                continue

            open_set.remove(current)

            if current == end_pos:
                return paths[current]

            neighbors = self.find_neighbors(grid, current)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                tentative_g_score = g_scores[current] + 1
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + self.heuristic(*neighbor)
                    
                    # If the neighbor isn't in the open set, add it.
                    if neighbor not in open_set:
                        open_set.add(neighbor)
                        open_list.put((f_scores[neighbor], self.heuristic(*neighbor), neighbor))

                    # Update the path leading to this neighbor.
                    paths[neighbor] = paths[current] + [neighbor]
                    
                visited.add(neighbor)

        return []

    def move(self, x, y, grid):
        path = self.find_path((x, y), grid)
        if len(path) > 1:
            next_step = path[1]
            return next_step
        return x, y

class BidirectionalAStar:
    def __init__(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.heuristic_cache = {}

    def heuristic(self, x, y):
        # Euclidean distance as heuristic
        return ((x - self.goal_x) ** 2 + (y - self.goal_y) ** 2) ** 0.5

    def find_neighbors(self, grid, x, y):
        neighbors = [(x + dx, y + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
        return [(nx, ny) for nx, ny in neighbors if grid.is_valid_move(nx, ny)]

    def move(self, x, y, grid):
        path = self.find_path((x, y), (self.goal_x, self.goal_y), grid)
        return path[1] if len(path) > 1 else (x, y)

    def find_path(self, start_pos, goal_pos, grid):
        start_open = []
        heapq.heappush(start_open, (0, start_pos))

        goal_open = []
        heapq.heappush(goal_open, (0, goal_pos))

        start_came_from = {start_pos: None}
        goal_came_from = {goal_pos: None}

        start_g_score = defaultdict(lambda: float('inf'))
        start_g_score[start_pos] = 0

        goal_g_score = defaultdict(lambda: float('inf'))
        goal_g_score[goal_pos] = 0

        while start_open and goal_open:
            # Expand the node with the lowest f-score from the start open list
            _, current_start = heapq.heappop(start_open)
            # Expand the node with the lowest f-score from the goal open list
            _, current_goal = heapq.heappop(goal_open)

            if current_start in goal_came_from:
                return self.reconstruct_path(start_came_from, goal_came_from, current_start)

            if current_goal in start_came_from:
                return self.reconstruct_path(start_came_from, goal_came_from, current_goal, reverse=True)

            for neighbor in self.find_neighbors(grid, *current_start):
                tentative_g_score = start_g_score[current_start] + 1
                if tentative_g_score < start_g_score[neighbor]:
                    start_came_from[neighbor] = current_start
                    start_g_score[neighbor] = tentative_g_score
                    heapq.heappush(start_open, (tentative_g_score + self.heuristic(*neighbor), neighbor))

            for neighbor in self.find_neighbors(grid, *current_goal):
                tentative_g_score = goal_g_score[current_goal] + 1
                if tentative_g_score < goal_g_score[neighbor]:
                    goal_came_from[neighbor] = current_goal
                    goal_g_score[neighbor] = tentative_g_score
                    heapq.heappush(goal_open, (tentative_g_score + self.heuristic(*neighbor), neighbor))

        return []

    def reconstruct_path(self, start_came_from, goal_came_from, meet_point, reverse=False):
        path = []

        # Reconstruct the path from the start to the meeting point
        current = meet_point
        while current:
            path.append(current)
            current = start_came_from[current]
        path.reverse()

        # Reconstruct the path from the goal to the meeting point
        current = meet_point
        if reverse:
            current = goal_came_from[current]
        while current:
            path.append(current)
            current = goal_came_from[current]

        return path

class ThetaStar:
    def __init__(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.heuristic_cache = {}

    def heuristic(self, x, y):
        if (x, y) not in self.heuristic_cache:
            self.heuristic_cache[(x, y)] = abs(x - self.goal_x) + abs(y - self.goal_y)
        return self.heuristic_cache[(x, y)]

    def line_of_sight(self, grid, start, end):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while x0 != x1 or y0 != y1:
            if not grid.is_valid_move(x0, y0):
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return True

    def find_neighbors(self, grid, pos, came_from):
        x, y = pos
        neighbors = [(x + dy, y + dx) for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
        valid_neighbors = [neighbor for neighbor in neighbors if grid.is_valid_move(*neighbor)]

        if pos in came_from:
            parent = came_from[pos]
            if self.line_of_sight(grid, parent, pos):
                valid_neighbors.append(parent)

        return valid_neighbors

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            next_node = came_from[current]
            x0, y0 = current
            x1, y1 = next_node
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            while x0 != x1 or y0 != y1:
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
                path.append((x0, y0))
            current = next_node
        path.reverse()
        return path

    def find_path(self, start_pos, grid):
        end_pos = (self.goal_x, self.goal_y)

        open_list = PriorityQueue()
        open_list.put((0, start_pos))
        visited = set()
        g_scores = {start_pos: 0}
        f_scores = {start_pos: self.heuristic(*start_pos)}
        came_from = {}

        while not open_list.empty():
            _, current = open_list.get()
            if current == end_pos:
                return self.reconstruct_path(came_from, current)

            visited.add(current)
            neighbors = self.find_neighbors(grid, current, came_from)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                tentative_g_score = g_scores[current] + (math.sqrt((neighbor[0] - current[0])**2 + (neighbor[1] - current[1])**2) if self.line_of_sight(grid, current, neighbor) else 1)
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + self.heuristic(*neighbor)
                    open_list.put((f_scores[neighbor], neighbor))

        return []

    def move(self, x, y, grid):
        path = self.find_path((x, y), grid)
        if len(path) > 1:
            next_step = path[1]
            return next_step
        return x, y


class MemoryEfficientAStar:
    def __init__(self, goal_x, goal_y, prune_threshold=1.5):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.heuristic_cache = {}
        self.prune_threshold = prune_threshold

    def heuristic(self, x, y):
        # Euclidean distance as heuristic
        return ((x - self.goal_x) ** 2 + (y - self.goal_y) ** 2) ** 0.5

    def find_neighbors(self, grid, x, y):
        neighbors = [(x + dx, y + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
        return [(nx, ny) for nx, ny in neighbors if grid.is_valid_move(nx, ny)]

    def move(self, x, y, grid):
        path = self.find_path((x, y), grid)
        return path[1] if len(path) > 1 else (x, y)

    def find_path(self, start_pos, grid):
        open_list = []
        heapq.heappush(open_list, (0, start_pos))

        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_pos] = 0

        f_score = defaultdict(lambda: float('inf'))
        f_score[start_pos] = self.heuristic(*start_pos)

        while open_list:
            current_f, current = heapq.heappop(open_list)

            if current == (self.goal_x, self.goal_y):
                return self.reconstruct_path(came_from, current)

            for neighbor in self.find_neighbors(grid, *current):
                tentative_g_score = g_score[current] + 1  # assuming cost of 1 for each move
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(*neighbor)
                    if neighbor not in [n for f, n in open_list]:
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))

            # Prune nodes from open list based on f-score threshold
            avg_f_score = sum(f for f, _ in open_list) / len(open_list)
            open_list = [node for node in open_list if node[0] <= avg_f_score * self.prune_threshold]

        return []

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]


class JPS:
    def __init__(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.heuristic_cache = {}

    def heuristic(self, x, y):
        if (x, y) not in self.heuristic_cache:
            self.heuristic_cache[(x, y)] = abs(x - self.goal_x) + abs(y - self.goal_y)
        return self.heuristic_cache[(x, y)]

    def find_neighbors(self, grid, pos):
        x, y = pos
        neighbors = [(x + dy, y + dx) for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
        valid_neighbors = [neighbor for neighbor in neighbors if grid.is_valid_move(*neighbor)]
        return valid_neighbors

    def jump(self, grid, current, direction):
        x, y = current
        dx, dy = direction

        new_x, new_y = x + dx, y + dy

        # If the next point isn't valid, return None
        if not grid.is_valid_move(new_x, new_y):
            return None

        # If the next point is the goal, return it
        if (new_x, new_y) == (self.goal_x, self.goal_y):
            return (new_x, new_y)

        # Check for forced neighbors
        # If we're moving diagonally
        if dx != 0 and dy != 0:
            if (grid.is_valid_move(new_x - dx, new_y) and not grid.is_valid_move(x - dx, y)) or \
               (grid.is_valid_move(new_x, new_y - dy) and not grid.is_valid_move(x, y - dy)):
                return (new_x, new_y)

        # If we're moving horizontally or vertically
        else:
            neighbors = self.find_neighbors(grid, (new_x, new_y))
            for neighbor in neighbors:
                if neighbor != (x, y) and abs(neighbor[0] - new_x) != abs(neighbor[1] - new_y):
                    return (new_x, new_y)

        # Continue in the current direction
        return self.jump(grid, (new_x, new_y), direction)

    def find_jump_points(self, grid, current):
        jump_points = []
        neighbors = self.find_neighbors(grid, current)
        for neighbor in neighbors:
            direction = (neighbor[0] - current[0], neighbor[1] - current[1])
            jump_point = self.jump(grid, current, direction)
            if jump_point:
                jump_points.append(jump_point)
        return jump_points

    def reconstruct_path(self, path):
        full_path = []
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            diff_x, diff_y = next_node[0] - current[0], next_node[1] - current[1]
            step_x, step_y = diff_x // abs(diff_x) if diff_x != 0 else 0, diff_y // abs(diff_y) if diff_y != 0 else 0

            while current != next_node:
                full_path.append(current)
                current = (current[0] + step_x, current[1] + step_y)
        full_path.append(path[-1])
        return full_path

    def find_path(self, start_pos, grid):
        end_pos = (self.goal_x, self.goal_y)

        open_list = PriorityQueue()
        open_list.put((0, (start_pos, [start_pos])))
        visited = set()

        g_scores = {start_pos: 0}
        f_scores = {start_pos: self.heuristic(*start_pos)}

        while not open_list.empty():
            _, (current, path) = open_list.get()

            # This is where we'll correct the visited set usage
            if current in visited:
                continue
            visited.add(current)

            if current == end_pos:
                return self.reconstruct_path(path)

            jump_points = self.find_jump_points(grid, current)
            for jump_point in jump_points:
                tentative_g_score = g_scores[current] + math.sqrt((jump_point[0] - current[0])**2 + (jump_point[1] - current[1])**2)
                if jump_point not in g_scores or tentative_g_score < g_scores[jump_point]:
                    g_scores[jump_point] = tentative_g_score
                    f_scores[jump_point] = tentative_g_score + self.heuristic(*jump_point)
                    open_list.put((f_scores[jump_point], (jump_point, path + [jump_point])))

        return []

    def move(self, x, y, grid):
        path = self.find_path((x, y), grid)
        if len(path) > 1:
            next_step = path[1]
            return next_step
        return x, y


class DFS:
    def __init__(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y

    def find_path(self, start_pos, grid):
        end_pos = (self.goal_x, self.goal_y)
        stack = [(start_pos, [start_pos])]
        visited = set()

        while stack:
            current, path = stack.pop()
            if current == end_pos:
                return path

            visited.add(current)
            neighbors = self.find_neighbors(grid, current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

        return []

    def find_neighbors(self, grid, pos):
        x, y = pos
        neighbors = [(x + dy, y + dx) for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
        valid_neighbors = [neighbor for neighbor in neighbors if grid.is_valid_move(*neighbor)]
        return valid_neighbors

    def move(self, x, y, grid):
        path = self.find_path((x, y), grid)
        if len(path) > 1:
            next_step = path[1]
            return next_step
        return x, y


class DStarLite:
    def __init__(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.heuristic_cache = {}
        self.INFINITY = float('inf')
        self.g = {}
        self.rhs = {}
        self.open_list = []

    def heuristic(self, x, y):
        if (x, y) not in self.heuristic_cache:
            self.heuristic_cache[(x, y)] = abs(x - self.goal_x) + abs(y - self.goal_y)
        return self.heuristic_cache[(x, y)]

    def find_neighbors(self, grid, pos):
        x, y = pos
        neighbors = [(x + dy, y + dx) for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
        valid_neighbors = [neighbor for neighbor in neighbors if grid.is_valid_move(*neighbor)]
        return valid_neighbors

    def calculate_key(self, node):
        g_value = self.g.get(node, self.INFINITY)
        rhs_value = self.rhs.get(node, self.INFINITY)
        return (min(g_value, rhs_value) + self.heuristic(*node), min(g_value, rhs_value))

    def update_vertex(self, node, grid):
        if node != (self.goal_x, self.goal_y):
            self.rhs[node] = min(self.g.get(neighbor, self.INFINITY) + 1 for neighbor in self.find_neighbors(grid, node))
        self.open_list = [item for item in self.open_list if item[1] != node]
        if self.g.get(node, self.INFINITY) != self.rhs.get(node, self.INFINITY):
            heapq.heappush(self.open_list, (self.calculate_key(node), node))

    def compute_shortest_path(self, grid):
        while self.open_list and (self.open_list[0][0] < self.calculate_key(self.open_list[0][1]) or self.rhs[self.open_list[0][1]] != self.g.get(self.open_list[0][1], self.INFINITY)):
            _, current_node = heapq.heappop(self.open_list)
            if self.g.get(current_node, self.INFINITY) > self.rhs[current_node]:
                self.g[current_node] = self.rhs[current_node]
                for neighbor in self.find_neighbors(grid, current_node):
                    self.update_vertex(neighbor, grid)
            else:
                self.g[current_node] = self.INFINITY
                self.update_vertex(current_node, grid)
                for neighbor in self.find_neighbors(grid, current_node):
                    self.update_vertex(neighbor, grid)

    def find_path(self, start_pos, grid):
        self.rhs[(self.goal_x, self.goal_y)] = 0

        if not self.open_list:
            heapq.heappush(self.open_list, (self.calculate_key((self.goal_x, self.goal_y)), (self.goal_x, self.goal_y)))

        self.compute_shortest_path(grid)

        path = [start_pos]
        current = start_pos
        while current != (self.goal_x, self.goal_y) and self.g.get(current, self.INFINITY) != self.INFINITY:
            current = min(self.find_neighbors(grid, current), key=lambda x: self.g.get(x, self.INFINITY))
            path.append(current)

        # If we couldn't reach the goal, return an empty path
        if current != (self.goal_x, self.goal_y):
            return []

        return path

    def move(self, x, y, grid):
        path = self.find_path((x, y), grid)
        if len(path) > 1:
            next_step = path[1]
            return next_step
        return x, y


class BeamSearch:
    def __init__(self, goal_x, goal_y, beam_width=5):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.beam_width = beam_width

    def heuristic(self, x, y):
        # Using Manhattan distance as heuristic
        return abs(x - self.goal_x) + abs(y - self.goal_y)

    def find_neighbors(self, grid, x, y):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if grid.is_valid_move(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def find_path(self, start_pos, grid):
        open_list = deque([start_pos])
        visited = set([start_pos])
        paths = {start_pos: [start_pos]}  # Initialize paths dictionary

        while open_list:
            # Expand the current level in the search tree
            current_level = []
            for _ in range(len(open_list)):
                current = open_list.popleft()
                for neighbor in self.find_neighbors(grid, *current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        current_path = paths[current] + [neighbor]
                        paths[neighbor] = current_path  # Update paths dictionary
                        current_level.append((self.heuristic(*neighbor), neighbor, current_path))
                        if neighbor == (self.goal_x, self.goal_y):
                            return current_path

            # Keep only the top beam_width nodes for the next level
            current_level.sort()
            for _, node, _ in current_level[:self.beam_width]:
                open_list.append(node)

        return []  # Return an empty path if the goal is not reached

    def move(self, x, y, grid):
        path = self.find_path((x, y), grid)
        if len(path) > 1:
            return path[1]
        return x, y


class Particle:
    def __init__(self, x, y, grid):
        self.x = x
        self.y = y
        self.velocity = (random.choice([-1, 1]), random.choice([-1, 1]))  # Randomize initial velocity
        self.p_best_x = x
        self.p_best_y = y
        self.grid = grid
        self.path = [(x, y)]

    def update_position(self):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        best_direction = None
        best_dot_product = -np.inf
        for direction in directions:
            new_x, new_y = self.x + direction[0], self.y + direction[1]
            if self.grid.is_valid_move(new_x, new_y) and (new_x, new_y) not in self.path:
                dot_product = np.dot(direction, self.velocity)
                if dot_product > best_dot_product:
                    best_dot_product = dot_product
                    best_direction = direction
        if best_direction is not None:
            self.x += best_direction[0]
            self.y += best_direction[1]
            self.path.append((self.x, self.y))

    def update_velocity(self, g_best_x, g_best_y, w=0.5, c1=2, c2=2, randomness=0.1):
        r1, r2 = np.random.rand(), np.random.rand()
        self.velocity = (w * self.velocity[0] + c1 * r1 * (self.p_best_x - self.x) +
                         c2 * r2 * (g_best_x - self.x) + randomness * np.random.randn(),
                         w * self.velocity[1] + c1 * r1 * (self.p_best_y - self.y) +
                         c2 * r2 * (g_best_y - self.y) + randomness * np.random.randn())
        # Normalize the velocity to keep its magnitude within a reasonable range
        magnitude = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if magnitude > 0:
            self.velocity = (self.velocity[0] / magnitude, self.velocity[1] / magnitude)

    def update_personal_best(self, goal_x, goal_y):
        if self.distance_to_goal(goal_x, goal_y) < self.distance_to_goal(self.p_best_x, self.p_best_y):
            self.p_best_x, self.p_best_y = self.x, self.y

    def distance_to_goal(self, goal_x, goal_y):
        return np.sqrt((goal_x - self.x) ** 2 + (goal_y - self.y) ** 2)


class PSO:
    def __init__(self, goal_x, goal_y, num_particles=10, max_iter=100):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.g_best_x = None
        self.g_best_y = None

    def distance_to_goal(self, x, y):
        return np.sqrt((self.goal_x - x) ** 2 + (self.goal_y - y) ** 2)

    def find_path(self, x, y, grid):
        particles = [Particle(x, y, grid) for _ in range(self.num_particles)]
        prev_g_best_x, prev_g_best_y = None, None
        stagnant_iters = 0  # Track number of iterations without change in g_best
        
        for _ in range(self.max_iter):
            for particle in particles:
                particle.update_position()
                particle.update_personal_best(self.goal_x, self.goal_y)
                if (self.g_best_x is None or self.g_best_y is None or
                        self.distance_to_goal(particle.x, particle.y) < self.distance_to_goal(self.g_best_x, self.g_best_y)):
                    self.g_best_x, self.g_best_y = particle.x, particle.y
            
            # Check for convergence
            if prev_g_best_x == self.g_best_x and prev_g_best_y == self.g_best_y:
                stagnant_iters += 1
            else:
                stagnant_iters = 0

            # Update previous g_best values
            prev_g_best_x, prev_g_best_y = self.g_best_x, self.g_best_y
            
            # Break loop if convergence is detected (e.g., no change for 10 iterations)
            if stagnant_iters >= 10:
                break

            # Update particle velocities
            for particle in particles:
                particle.update_velocity(self.g_best_x, self.g_best_y)
        
        return particles[np.argmin([p.distance_to_goal(self.goal_x, self.goal_y) for p in particles])].path

    def move(self, x, y, grid):
        path = self.find_path(x, y, grid)
        if len(path) > 1:
            next_step = path[1]
            return next_step
        return x, y


class Simulation:
    def __init__(self, grid, agents, goal_x, goal_y, max_iterations=1000):
        self.grid = grid
        self.agents = agents
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.max_iterations = max_iterations
        self.iterations = 0
        self.agents_reached_goal = set()

        self.cell_size = 40
        self.window_width = self.grid.width * self.cell_size
        self.window_height = self.grid.height * self.cell_size

        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.yellow = (255, 255, 0)
        self.fps = 10

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Grid-Based Simulation")
        self.clock = pygame.time.Clock()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def draw_grid(self):
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell_color = self.white
                if self.grid.grid[y][x] == CellType.OBSTACLE:
                    cell_color = self.black
                elif self.grid.grid[y][x] == CellType.GOAL:
                    cell_color = self.yellow

                pygame.draw.rect(self.screen, cell_color, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size), 0)
                pygame.draw.rect(self.screen, self.black, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size), 1)

    def draw_agents(self):
        for idx, agent in enumerate(self.agents):
            agent_color = self.green if idx % 2 == 0 else self.blue
            pygame.draw.circle(self.screen, agent_color, (agent.x * self.cell_size + self.cell_size // 2, agent.y * self.cell_size + self.cell_size // 2), self.cell_size // 2)

    def update(self):
        for idx, agent in enumerate(self.agents):
            if idx in self.agents_reached_goal:
                continue

            agent.move(self.grid)

            if (agent.x, agent.y) == (self.goal_x, self.goal_y):
                self.agents_reached_goal.add(idx)
                print(f"Agent {idx} reached the goal at iteration {self.iterations}")

        self.iterations += 1

    def run(self):
        running = True
        while running and self.iterations < self.max_iterations and len(self.agents_reached_goal) < len(self.agents):
            self.screen.fill(self.white)
            self.draw_grid()
            self.draw_agents()
            pygame.display.flip()

            self.update()
            self.handle_events()

            self.clock.tick(self.fps)

        if len(self.agents_reached_goal) == len(self.agents):
            print("All agents reached the goal.")
        else:
            print("Simulation ended. Not all agents reached the goal.")

        pygame.quit()


# Create and populate the grid environment
width, height = 10, 10
grid = Grid(width, height)
start_x, start_y = 0, 0
goal_x, goal_y = 9, 9
grid.generate_maze(start_x, start_y, goal_x, goal_y)

# Create agents
agents = [
    #Agent(start_x, start_y, AStar(goal_x, goal_y)),
    #Agent(start_x, start_y, BidirectionalAStar(goal_x, goal_y)),
    #Agent(start_x, start_y, ThetaStar(goal_x, goal_y)),
    #Agent(start_x, start_y, MemoryEfficientAStar(goal_x, goal_y)),
    #Agent(start_x, start_y, JPS(goal_x, goal_y)),
    #Agent(start_x, start_y, DFS(goal_x, goal_y)),
    #Agent(start_x, start_y, DStarLite(goal_x, goal_y)),
    Agent(start_x, start_y, BeamSearch(goal_x, goal_y)),
    #Agent(start_x, start_y, PSO(goal_x, goal_y)),
]

# Run the simulation
simulation = Simulation(grid, agents, goal_x, goal_y)
simulation.run()



import random
import time


def run_pathfinding_comparison(grid_size, algorithms, num_runs):
    # Dictionary to store cumulative times for each algorithm
    cumulative_times = {algo.__name__: 0. for algo in algorithms}

    for _ in range(num_runs):
        # Start and goal positions relative to the grid size
        start_x, start_y = 0, 0
        goal_x, goal_y = grid_size - 1, grid_size - 1

        # Create and populate the grid environment for each run
        grid = Grid(grid_size, grid_size)
        grid.generate_maze(start_x, start_y, goal_x, goal_y)

        for algo in algorithms:
            # Create an agent with the specified algorithm
            agent = Agent(start_x, start_y, algo(goal_x, goal_y))

            # Start the timer when the agent starts moving
            start_time = time.time()

            # Simulate the agent until it reaches the goal
            while (agent.x, agent.y) != (goal_x, goal_y):
                agent.move(grid)

            # Stop the timer when the agent reaches the goal
            end_time = time.time()

            # Accumulate the time taken
            cumulative_times[algo.__name__] += end_time - start_time

    # Average the times over all runs
    average_times = {algo: total_time / num_runs for algo, total_time in cumulative_times.items()}
    
    return average_times

# Example usage
# algorithms = [AStar, BidirectionalAStar, ThetaStar, MemoryEfficientAStar, JPS, DFS, DStarLite]
# grid_size = 99  # Size of the grid
# num_runs = 1  # Number of runs for each algorithm
# average_times = run_pathfinding_comparison(grid_size, algorithms, num_runs)
# print(average_times)

"""
1 random grids of size 99
{'AStar': 26.696054935455322, 'BidirectionalAStar': 11.132912874221802, 'ThetaStar': 21.85900616645813, 'MemoryEfficientAStar': 13.792948484420776, 'JPS': 38.1959342956543, 'DFS': 41.405953884124756, 'DStarLite': 3.6806371212005615}
"""
"""
100 random grids of size 9
{'AStar': 0.007079935073852539, 'BidirectionalAStar': 0.004592006206512451, 'ThetaStar': 0.009288866519927979, 'MemoryEfficientAStar': 0.005601677894592285, 'JPS': 0.014568932056427002, 'DFS': 0.005128917694091797, 'DStarLite': 0.0036286449432373045}
"""

# http://www.codenamepandey.com/movementalgo
# https://en.wikipedia.org/wiki/Multi-agent_pathfinding
# 1 change to grid.get_cost of that cell and neighbor
# https://gamedev.stackexchange.com/questions/141688/how-to-optimize-pathfinding-on-a-very-large-dynamic-2d-grid
# https://grail.cs.washington.edu/projects/crowd-flows/78-treuille.pdf
# PSO also draw all particles
# memorization to avoid recalculation

"""
Multi-agent pathfinding
Main article: Multi-agent pathfinding
Multi-agent pathfinding is to find the paths for multiple agents from their current locations to their target locations without colliding with each other, while at the same time optimizing a cost function, such as the sum of the path lengths of all agents. It is a generalization of pathfinding. Many multi-agent pathfinding algorithms are generalized from A*, or based on reduction to other well studied problems such as integer linear programming.[10] However, such algorithms are typically incomplete; in other words, not proven to produce a solution within polynomial time. A different category of algorithms sacrifice optimality for performance by either making use of known navigation patterns (such as traffic flow) or the topology of the problem space.[11]
"""

"""
Johnson算法

Johnson算法是一種用於在加權有向圖中找到所有節點對之間最短路徑的單源最短路徑算法，由Donald Johnson於1977年提出。該算法是一種基於Bellman-Ford算法和Dijkstra算法的改進算法，它的時間複雜度為O(V^2logV + VE)，其中V是節點數，E是邊數。

Johnson算法的主要思想是先通過將每個節點的權重重新賦值，使得圖中不存在負權邊，然後再運用Dijkstra算法來計算所有節點對之間的最短路徑。為了使圖中不存在負權邊，Johnson算法使用了一種稱為“SPFA”的算法來計算每個節點的權重，這個算法類似於Bellman-Ford算法，但是它具有更高的效率。

在Johnson算法中，每個節點的權重被重新賦值為其到所有其他節點的最短路徑中最小的那個值。這個過程可以通過對圖中每個節點運行一遍SPFA算法來完成。然後，對於每個節點，我們可以通過運行Dijkstra算法來計算它到所有其他節點的最短路徑。最後，所有節點對之間的最短路徑可以通過將每個節點的Dijkstra算法計算結果加上它們之間的權重來得到。

雖然Johnson算法比Dijkstra算法和Bellman-Ford算法更複雜，但是它可以處理帶有負權邊的圖，並且在某些情況下比Floyd-Warshall算法更快。

Fredman和Tarjan的算法

Fredman和Tarjan在1984年的論文中提出了一種新的算法，用於在有向圖中解決帶非負邊權重的單源最短路問題。該算法的時間複雜度為O（m log n），其中n是圖中的頂點數，m是邊的數量。

該算法基於Dijkstra算法，該算法維護一組已訪問的頂點和一個待探索的頂點的優先級隊列。 Fredman和Tarjan算法的主要思想是將頂點分為兩個集合，活動和非活動，並根據它們到源的潛在距離維護一個活動頂點的堆。

在算法的每一步中，從活動頂點堆中刪除具有最小潛在距離的頂點，並將其添加到已訪問的頂點集合中。然後，對於每條外向邊，如果必要，則更新相應目標頂點的潛在距離，並且如果它們尚未被訪問，則將它們添加到活動頂點堆中。

潛在距離用於估計源到頂點的實際距離。它們對於源頂點初始化為零，並在從活動頂點堆中刪除頂點時進行更新。頂點的潛在距離是其任何入邊的最小潛在距離加上入邊的權重。

該算法的正確性基於已訪問頂點的潛在距離始終等於它們到源的實際距離，而活動頂點的潛在距離始終是它們到源的實際距離的下限。因此，當活動頂點堆為空時，算法可以終止，所有頂點的潛在距離給出源到它們的實際距離。

總的來說，Fredman和Tarjan的算法在時間複雜度方面相對於Dijkstra算法提供了顯著的改進，但需要更多的內存來維護活動頂點的堆。

Thorup 演算法

Thorup 在 1999 年提出了一個單源最短路徑演算法，稱為 Thorup 演算法。這個演算法的時間複雜度是 O(m log n)，其中 n 是圖中節點的數量，m 是邊的數量。

Thorup 演算法的主要思想是將圖分成多個層次，每個層次包含一組節點，並且在這些節點之間只存在一定的關係。在這個演算法中，Thorup 使用了一個稱為“漸進式疊代縮減”的技術，通過反覆地將圖分解成更小的子圖，最終得到一個只有兩個節點的子圖，並在此基礎上求出最短路徑。

Thorup 演算法的優點是具有良好的理論基礎，並且能夠處理一些具有特殊結構的圖，例如稠密圖和稀疏圖。然而，實際應用中，由於其實現複雜度較高，因此可能不如其他較簡單的演算法，例如 Dijkstra 算法和 Bellman-Ford 算法，具有更好的實際效率。

Gabow's 演算法

Gabow's 演算法是一種用於在有向圖中解決單源最短路徑問題的算法。該算法由Harold N. Gabow於1985年提出，它的時間複雜度為O(mlogn)，其中n是節點數，m是邊數。

Gabow's 演算法是一種基於Dijkstra算法的改進算法。它使用了一種稱為“桶優化”的技術，該技術可以顯著減少Dijkstra算法中需要處理的節點數。

在Gabow's 演算法中，節點被分為三個類別：已訪問的節點、待訪問的節點和未訪問的節點。算法從源節點開始，將源節點標記為已訪問的節點，並將其相鄰的節點加入待訪問的節點中。

然後，算法從待訪問節點中選取一個與源節點距離最短的節點，並將其標記為已訪問的節點。然後，算法將該節點的所有未訪問的相鄰節點加入待訪問節點中，並根據其距離從小到大放入不同的桶中。

在接下來的每一步中，算法從所有未訪問節點中選取與源節點距離最短的節點，並將其標記為已訪問的節點。然後，算法將該節點的所有未訪問的相鄰節點加入待訪問節點中，並將其根據距離放入不同的桶中。

當算法完成時，所有節點的最短路徑都已計算出來。

Gabow's 演算法的優點是能夠處理稀疏圖和密集圖，並且其時間複雜度相對較低。但是，該算法需要額外的內存來存儲節點和桶，並且不如其他一些算法，例如Dijkstra算法和Bellman-Ford算法，易於實現。
"""

"""
這些算法都是用來解決圖中最短路徑問題的，下面是它們各自的優缺點和適用情況：

1. Dijkstra算法：
   優點：對於邊權值非負的圖來說，是最快的單源最短路徑算法，時間複雜度為O(E+VlogV)。
   缺點：對於邊權值為負的圖，不能處理；需要額外的數據結構支持（如堆）。
   適用情況：邊權值非負的圖。

2. Bellman-Ford算法：
   優點：能夠處理邊權值為負的圖，也可以檢測負權環；不需要額外的數據結構支持。
   缺點：時間複雜度為O(EV)，比Dijkstra算法慢。
   適用情況：邊權值可以為負的圖，或者需要檢測負權環的情況。

3. Shortest Path Faster算法（SPFA）：
   優點：對於一般的稠密圖（即E接近V^2）來說，比Dijkstra算法快；可以處理負權邊。
   缺點：可能會陷入死循環，需要進行優化；時間複雜度最壞為O(EV)。
   適用情況：邊權值可以為負的圖，或者是稠密圖。

4. Floyd-Warshall算法：
   優點：可以求解任意兩點間的最短路徑；不受邊權值正負的限制。
   缺點：時間複雜度為O(V^3)，空間複雜度也較大。
    適用情況：圖中邊權值可以為正負的情況，求解任意兩點間的最短路徑。

5. Johnson算法：
   優點：對於稀疏圖，比Floyd-Warshall算法更快；可以處理負權邊。
   缺點：需要先求解最短路徑的估計值，複雜度為O(EVlogV)，然後再進行Dijkstra算法，總的時間複雜度為O(EVlogV)。
    適用情況：邊權值可以為負的圖，或者是稀疏圖。
"""

"""
https://en.wikipedia.org/wiki/DARPA_Grand_Challenge#Technology
"""

"""
Ma, H., & Koenig, S. (2016). Optimal target assignment and path finding for teams of agents. arXiv preprint arXiv:1612.05693.
"""

"""
Stern, R., Sturtevant, N., Felner, A., Koenig, S., Ma, H., Walker, T., ... & Boyarski, E. (2019). Multi-agent pathfinding: Definitions, variants, and benchmarks (PDF). arXiv preprint arXiv:1906.08291.
"""

"""
"Multi-level path planning" or "hierarchical motion planning" is an approach to motion planning that involves breaking down a complex task into smaller, more manageable subtasks. This is achieved by dividing the environment or workspace into clusters, where each cluster represents a region of the environment that can be navigated relatively easily.

On the high-level layer, a path is planned between the clusters, which represents the overall motion plan. This plan provides a rough guidance for navigating between the clusters while avoiding obstacles and minimizing the distance traveled. 

On the low-level layer, a second path is planned within each cluster to navigate the robot through the cluster while avoiding any obstacles within that cluster. This plan is based on the high-level plan and provides the detailed guidance for navigating through the local environment.

The advantage of using a hierarchical motion planning approach is that it can reduce the number of nodes that need to be considered during planning, thereby improving the computational efficiency of the algorithm. Additionally, it can help to simplify the planning problem by breaking it down into smaller, more manageable subtasks.

However, implementing a hierarchical path planner can be challenging, as it requires developing algorithms for both the high-level and low-level planning tasks, as well as developing a mechanism for integrating the two plans to ensure that they are coherent and feasible. Moreover, designing an appropriate cluster hierarchy that accurately captures the environment's structure is often not trivial, and it requires expertise in both the domain of motion planning and the specific application domain.
"""

"""
A quadtree is a tree data structure that can be used to represent two-dimensional space and partition it into smaller squares or rectangles. It is commonly used for spatial indexing, including in computer graphics, image processing, and geographical information systems. One application of quadtree is hierarchical pathfinding, where the quadtree represents a map or terrain and the pathfinding algorithm searches for a path between two points.

The hierarchical pathfinding algorithm can start by searching for a path between the start and end points on the highest level of the quadtree. If the path is blocked by an obstacle or the terrain is difficult to traverse, the algorithm can move down to a lower level of the quadtree and search for a path between smaller areas of the map. Each leaf node of the quadtree represents a small area of the map and contains information about the terrain or obstacles in that area. This process can be repeated until the algorithm reaches the lowest level of the quadtree, where it can perform a detailed search to find the exact path between the start and end points.

Quadtree operations can also be used to solve other problems, such as quickly determining the nearest neighbor of a point in a set of points or efficiently storing and querying spatial data. By using a quadtree for hierarchical pathfinding, the algorithm can efficiently search large areas of the map and avoid the need to perform detailed searches on areas of the map that are easy to traverse or do not contain obstacles, significantly reducing the computational cost of the pathfinding algorithm and improving its performance.
"""



"""
### Hierarchical A* (HPA*)

HPA* is an adaptation of A* that uses a hierarchy of abstraction levels to simplify pathfinding. To implement HPA*:

1. **Map Abstraction**: Divide the map into several hierarchical levels. At each level, the map is more abstracted, meaning it has fewer and larger nodes.

2. **Inter-Level Pathfinding**: Implement pathfinding at each level of the hierarchy. At higher levels, pathfinding covers larger distances with less detail. At lower levels, pathfinding is more detailed but covers shorter distances.

3. **Path Refinement**: Once a path is found at a higher abstraction level, refine it at the lower levels to get a detailed and accurate path.

4. **Dynamic Updates**: For environments that change, you will need to update the hierarchical representation dynamically, which could involve recalculating the abstraction at various levels.


Implementing Hierarchical Pathfinding A* (HPA*) in your existing simulation requires significant modifications and additions. I'll provide an extended code snippet that includes the necessary classes and modifications to integrate HPA* into your simulation. This will involve creating abstract nodes and edges, pathfinding at multiple levels of abstraction, and refining paths to work with your grid.

### Step 1: Abstraction Layer and Graph

These classes create an abstract representation of the grid, dividing it into clusters and creating nodes and edges between them.

```python
class AbstractNode:
    def __init__(self, id, center):
        self.id = id
        self.center = center

class AbstractEdge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

class AbstractGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(list)

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, node1, node2):
        edge = AbstractEdge(node1, node2)
        self.edges[node1.id].append(edge)
        self.edges[node2.id].append(edge)

class AbstractionLayer:
    def __init__(self, grid, cluster_size):
        self.grid = grid
        self.cluster_size = cluster_size
        self.abstract_graph = AbstractGraph()

    def generate_abstraction(self):
        # Implement logic to generate nodes and edges
        pass
```

### Step 2: Hierarchical A* Pathfinding

This class uses the abstract graphs to find paths at different abstraction levels.

```python
class HPAStar:
    def __init__(self, abstraction_layer):
        self.abstraction_layer = abstraction_layer

    def find_path(self, start, goal):
        # Implement HPA* algorithm here
        pass

    def refine_path(self, high_level_path):
        # Refine the high-level path
        pass
```

### Step 3: Integration into the Agent

Modify the `Agent` class to use HPA* for pathfinding.

```python
class Agent:
    def __init__(self, x, y, pathfinding_algorithm):
        self.x = x
        self.y = y
        self.pathfinding_algorithm = pathfinding_algorithm

    def move(self, grid):
        # Adjust the move function to use HPA* for pathfinding
        pass
```

### Step 4: Setting up the Simulation

Create abstraction layers and initialize HPA* for your agents.

```python
# Create abstraction layers
abstraction_layer = AbstractionLayer(grid, cluster_size=5)
abstraction_layer.generate_abstraction()

# Initialize HPA* with the abstraction layers
hpa_star = HPAStar(abstraction_layer)

# Create agents with HPA*
agents = [Agent(0, 0, hpa_star)]

# Rest of the simulation setup...
```

### Additional Considerations

1. **Abstraction Generation**: Implement the `generate_abstraction` method in `AbstractionLayer` to create nodes and edges based on your grid. This could involve dividing the grid into clusters and connecting adjacent clusters.

2. **Pathfinding in HPAStar**: The `find_path` method in `HPAStar` should find paths using the abstract graphs and then refine them for the actual grid.

3. **Path Refinement**: The `refine_path` method should convert the abstract path into a detailed path that can be followed in the grid.

4. **Dynamic Environment Handling**: If your environment changes, update the abstraction layers accordingly.

This implementation provides a framework for HPA* in your simulation. The detailed implementation of methods like `generate_abstraction`, `find_path`, and `refine_path` will depend on the specifics of your grid and the abstraction strategy you choose.
"""

"""
### Field D*

Field D* is an advanced pathfinding algorithm that is an extension of the D* algorithm, optimized for continuous domains and often used in robotics. To implement Field D*:

1. **Continuous Space Representation**: Unlike traditional grid-based maps, Field D* works in a continuous space. You would need to represent the environment as a continuous field, possibly using a graph structure where nodes represent locations in the space and edges represent navigable paths.

2. **Dynamic Environment Handling**: Field D* is designed to handle dynamic changes in the environment. This requires updating the path as new information becomes available, re-planning from the current location to the goal.

3. **Path Smoothing**: Since Field D* does not constrain paths to grid edges, it can result in paths with unnecessary turns. Implement a path smoothing algorithm to create more natural and efficient paths.
"""

"""
This excerpt appears to be from a scholarly paper discussing advancements in heuristic planners, specifically focusing on various algorithms used for pathfinding and motion planning in robotics and other computational fields. The algorithms mentioned include:

1. **A*** Variants:
   - **Field D* (FD*)**: Like Theta*, it doesn't constrain paths to grid edges but may include unnecessary turns.
   - **MEA* (Memory-Efficient A*)**: Incorporates a pruning process, offering better performance, lower memory use, and reduced computational time compared to A* and HPA*.
   - **HPA* (Hierarchical A*)**: Another variant using a pruning process.

2. **Sampling-Based Algorithms**:
   - **RRT (Rapidly-Exploring Random Tree)**: Suitable for high-dimensional complex problems.
   - **RRT***: Asymptotically optimal, discovering initial paths quickly and enhancing them in consecutive iterations. However, it has a slower convergence rate than basic RRT.
   - **RRT*-Smart**: An improvement on RRT*, achieving near-optimal paths more quickly.
   - **RRT*-AB (RRT*-Adjustable Bounds)**: A variation that converges rapidly to optimal or near-optimal paths compared to other RRT*-based approaches.

The paper contrasts these algorithms, focusing on MEA* in comparison with prominent sampling-based and grid-based algorithms. MEA* is highlighted for its shorter path generation, improved execution time, and lower memory requirements, particularly effective for 2D, off-line applications with constrained resources. However, in high-dimensional problems, RRT*-AB is expected to outperform MEA* due to its suitability for complex environments. The paper also mentions future research directions, including motion planning with RRT*-AB in 3D space for 6 DOF robotic arms and the use of fuzzy logic-based navigation for cooperative mobile robots in 3D environments.
"""

"""
Several planners have been used for mobile robots such as potential field, visibility graph, evolutionary meta-heuristic methods, sampling-based methods, and grid-based methods [3]. Each of them has their own advantages and disadvantages. Classic methods such as potential field and visibility graphs are complex, and computationally expensive to deal with real-time applications and high-dimensional problems. Nature-inspired meta-heuristic approaches such as Genetic Algorithm (GA) [10], and Artificial Bee Colony (ABC) algorithm [11] are suitable for the optimization of multi-objective planning problems. A major drawback of these approaches is pre-mature convergence, trapping in a local optimum, high computational cost, and complex mapping of data [3,11,12,13]. Recently, reinforcement learning has also emerged, but it is more suitable for robots learning new skills than for path-planning applications [14,15]. Sampling-based Planning (SBP) approaches such as RRT* (Rapidly exploring Random Tree Star) [16] are successful for high-dimensional complex problems [3]. A major limitation of sampling-based algorithms is their slow convergence rate [3,13]. RRT*-AB [12] is a recent sampling-based planner which has resolved this issue and has improved convergence rate as compared to other SBP variants [12]. Grid-based methods are another class of planning technique applicable to low-dimensional space. A* [17] is a popular grid-based algorithm and is usually preferred to solve planning problems of mobile robots in low dimensions [1,4,18,19]. However, A* does not always find the shortest path because the path is constrained to grid edges. Its memory requirements also expand vigorously for complex problems [4,6]. Memory-Efficient A* (MEA*) [20] is a variant of A* which has addressed these limitations and its performance is also comparable to A* as compared to other variants [20]. However, there is always a trade-off between processing time and robustness.
"""

"""
https://www.mdpi.com/2073-8994/11/7/945
3.1. MEA* Algorithm
3.2. RRT*-AB Algorithm

https://journals.sagepub.com/doi/10.5772/56718

EB-RRT* based Navigation Algorithm for UAV

https://www.semanticscholar.org/paper/RT-RRT*%3A-a-real-time-path-planning-algorithm-based-Naderi-Rajam%C3%A4ki/a28ff0c2890fb178991bedf4f7ea66537c47ff24

https://www.mdpi.com/2076-3417/10/4/1381

https://www.semanticscholar.org/paper/UAV-path-re-planning-based-on-improved-RRT-in-Meng-Qing/171bdecb09f27fb5716ef09523e9db72b2a6e071

https://vhartmann.com/strrt/
"""

"""
Hybrid A*
CHOMP: Covariant Hamiltonian Optimization for Motion Planning
TrajOpt: Motion Planning with Sequential ConvexOptimization and Convex Collision Checking
Artificial Potential Field
"""

"""
# Beam search, a heuristic search algorithm, offers robust applications in a zombie apocalypse simulation, enhancing the depth and complexity of the scenario. Here's an integrated and detailed look at how beam search can be employed across different aspects of such a simulation:
# 1. **Pathfinding and Navigation**:
#    - **For Survivors**: Beam search calculates the safest and most efficient paths for survivors to reach destinations like safe zones and resource caches. It evaluates multiple potential paths simultaneously, considering factors such as the shortest distance, zombie density, proximity to resources, and terrain challenges.
#    - **For Zombies**: The algorithm simulates intelligent movement patterns for zombies, determining paths that maximize encounters with survivors and spread of infection. Factors like noise, light, and recent survivor activity are considered, enhancing the unpredictability and threat posed by zombies.
# 2. **Resource Optimization**:
#    - Beam search assists in making critical decisions regarding resource allocation and optimization. It explores different strategies for distributing scarce resources like food, water, weapons, and medical supplies among survivors. The algorithm weighs the urgency of needs, health status, and long-term sustainability, helping to choose the best locations for scavenging and the most efficient use of available resources.
# 3. **Survivors' Group Formation**:
#    - The algorithm aids in forming survivor groups by analyzing various factors such as skills, compatibility, trust, and survivability. It explores different combinations of individuals and their potential synergies, helping to create teams that maximize the chances of survival, success in scavenging missions, base building, or confrontations with zombies.
# 4. **Decision-making and Strategy**:
#    - Beam search plays a crucial role in evaluating different options and potential outcomes for critical decisions. This includes choosing the best locations for establishing safe zones, selecting optimal defense strategies, and planning rescue operations. It helps survivors make informed decisions that increase their survival chances, considering a range of possible choices and their consequences.
# 5. **Zombie Movement and Spread Patterns**:
#    - The algorithm can predict the spread of the zombie outbreak across the map by analyzing human movement patterns, geographical barriers, and environmental factors. This helps create dynamic scenarios where the level of threat changes over time, making the simulation more realistic and challenging.
# 6. **Escape Route Optimization**:
#    - In high-risk situations where quick evacuation is necessary, beam search evaluates multiple escape routes. It considers factors like safety, distance, and the probability of encountering zombies, aiding survivors trapped in dangerous areas to make swift and strategic escape decisions.
# 7. **Long-Term Survival Planning**:
#    - Beam search assists in planning for the sustainable growth and development of survivor communities. It explores scenarios for choosing settlement locations, resource cultivation strategies, and establishing defensive structures, considering long-term factors such as resource availability, potential threats, and population dynamics.
# By integrating beam search in these detailed applications, a zombie apocalypse simulation becomes highly dynamic and realistically challenging. The algorithm's capability to quickly sift through numerous possibilities and identify promising options is particularly valuable in the unpredictable and high-stakes context of a zombie apocalypse. This approach not only enriches the gameplay or theoretical modeling experience but also provides deep insights into survival strategies and decision-making under extreme conditions.

# Beam Stack Search: Beam stack search combines beam search with depth-first search (DFS). It maintains a stack of active search paths (beams) instead of a single beam. It starts by expanding the beam in a depth-first manner, while also keeping track of the best states found so far. This variant allows for deeper exploration of the search space while still benefiting from the pruning capabilities of beam search. It is an anytime algorithm that can find sub-optimal solutions quickly and gradually converge to an optimal solution. This allows for faster identification of good solutions compared to pure depth-first search, while also exploring a wider range of possibilities than beam search alone. Use beam stack search when you want to balance the benefits of depth-first search and beam search. It is useful when you need to explore the search space more deeply while still benefiting from the pruning capabilities of beam search. Beam stack search is an anytime algorithm that can quickly find sub-optimal solutions and gradually converge to an optimal solution.
# Depth-First Beam Search: Depth-first beam search is another combination of beam search and depth-first search. It uses a stack to maintain a beam of active search paths, similar to beam stack search. However, instead of expanding all beams at each level simultaneously, it explores a single beam in a depth-first manner but limits the depth of the search to a specific number of steps. This approach allows for deeper exploration while retaining the memory efficiency of beam search. Choose depth-first beam search when you want to explore the search space more deeply than traditional beam search but still maintain memory efficiency. It is suitable when you have a specific limit on the depth of the search and want to prioritize depth-first exploration over breadth-first exploration.
# Beam Search using Limited Discrepancy Backtracking (BULB): BULB is a variant of beam search that incorporates limited discrepancy search to guide the search. The limited discrepancy heuristic is a measure of how different a state is from the goal state. Limited discrepancy search tries to expand states that are close to the goal state, but also introduces controlled randomness during backtracking to explore different search paths. BULB combines beam search with limited discrepancy backtracking, allowing it to quickly find good sub-optimal solutions using beam search, and then continue searching for improved solutions until convergence to an optimal solution. BULB is suitable when you want to combine the benefits of beam search and limited discrepancy search. It is useful when you want to quickly find good sub-optimal solutions using beam search and then continue searching for improved solutions until convergence to an optimal solution. BULB is effective when you have a heuristic measure of discrepancy that can guide the search.
# Local Beam Search: Local beam search is a variant of beam search that is often used in the context of local search algorithms. It begins by randomly generating a predetermined number of states and considers the same number of new states at each level of the search tree among all possible successors of the current states. This variant aims to reach a goal by exploring a limited number of promising states at each step. Use local beam search when you are applying beam search in the context of local search algorithms. It is useful when you want to explore a limited number of promising states at each step, potentially leading to a solution in scenarios where the search space is large.
# Stochastic Beam Search: Stochastic beam search is a modification of local beam search that addresses the issue of getting stuck in local maxima. Instead of selecting the next beta states deterministically, stochastic beam search introduces randomness. The choice of the next beta states is made probabilistically, with the probability dependent on the heuristic evaluation of the states. This randomness allows for a more diverse exploration of the search space, potentially leading to better solutions by avoiding being trapped in local optima. Choose stochastic beam search when you want to address the issue of getting stuck in local optima. It is suitable when you want to introduce randomness to the selection of next states, allowing for more diverse exploration of the search space and potentially finding better solutions.
# Flexible Beam Search: Flexible beam search allows the user to specify the number of states (beams) to expand at each level of the search tree. This variant is useful when the user possesses domain knowledge and wants to control the exploration of the search space accordingly. Use flexible beam search when you have domain knowledge and want to control the exploration of the search space by specifying the number of states (beams) to expand at each level. It allows for customization based on your specific requirements and understanding of the problem.
# Recovery Beam Search: Recovery beam search enables the search algorithm to recover from errors or incorrect decisions. If the search makes a mistake, it can backtrack to a previous state and try a different path. Recovery beam search is beneficial in scenarios where the search space is vast and errors are likely to occur, allowing for more robust exploration of the search space. Choose recovery beam search when you want to build robustness into the search algorithm by allowing it to recover from errors or incorrect decisions. It is useful when the search space is vast and errors are likely to occur, enabling the algorithm to backtrack and explore alternative paths.
# Consider the characteristics of your problem, such as the size of the search space, the trade-off between exploration and exploitation, and the availability of heuristic information. These factors will help you determine the most suitable variant of beam search for your specific application.
"""
