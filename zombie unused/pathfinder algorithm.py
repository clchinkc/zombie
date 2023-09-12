

"""
There are many ways to implement more complex algorithms for controlling the movements of people and zombies in a simulation. Here are a few examples of more advanced approaches you could consider:

Pathfinding algorithms: You could use algorithms such as A* or Dijkstra's algorithm to allow people and zombies to intelligently navigate the school layout and avoid obstacles. This could be useful for simulating more realistic or strategic movements, such as people trying to find the best route to escape the school or zombies trying to track down nearby survivors.

Flocking algorithms: You could use algorithms such as Boids or Reynolds' steering behaviors to simulate the collective movements of groups of people or zombies. This could be used to model the behavior of crowds or hordes, and could help to create more realistic and dynamic simulations.

Machine learning algorithms: You could use machine learning techniques such as reinforcement learning or decision trees to train people and zombies to make more intelligent decisions about their movements. This could allow the simulation to adapt and improve over time, and could potentially enable the people and zombies to learn from their experiences and make more effective decisions in the future.
"""


import heapq
import math
import random
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

        for y in range(self.height):
            for x in range(self.width):
                if (self.height % 2 == 0 and y == self.height - 1) or (self.width % 2 == 0 and x == self.width - 1):
                    self.grid[y][x] = CellType.EMPTY

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

        for y in range(self.height):
            for x in range(self.width):
                if x % 2 == 0 or y % 2 == 0:
                    self.grid[y][x] = CellType.OBSTACLE

        start_cell = (start_x, start_y)
        visited = {start_cell}
        frontier = get_neighbors(start_x, start_y)

        while frontier:
            current_cell = frontier.pop(np.random.randint(len(frontier)))
            x, y = current_cell
            connecting_neighbors = [
                (nx, ny) for nx, ny in get_neighbors(x, y) if (nx, ny) in visited
            ]
            if connecting_neighbors:
                cx, cy = connecting_neighbors[np.random.randint(len(connecting_neighbors))]
                remove_wall(x, y, cx, cy)
                visited.add(current_cell)
                frontier.extend([n for n in get_neighbors(x, y) if n not in visited and n not in frontier])

        for y in range(self.height):
            for x in range(self.width):
                if (self.height % 2 == 0 and y == self.height - 1 and end_y == self.height - 1) or (
                        self.width % 2 == 0 and x == self.width - 1 and end_x == self.width - 1):
                    self.grid[y][x] = CellType.EMPTY

        self.set_goal(end_x, end_y)

# or wave function collapse
# Greedy Best-First Search
# Bidirectional Search
# Recursive Division
# Eller's Algorithm
# Kruskal's Algorithm
# Prim's Algorithm
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
        self.velocity = (self.velocity[0]/magnitude, self.velocity[1]/magnitude)

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
        for _ in range(self.max_iter):
            for particle in particles:
                particle.update_position()
                particle.update_personal_best(self.goal_x, self.goal_y)
                if (self.g_best_x is None or self.g_best_y is None or
                        self.distance_to_goal(particle.x, particle.y) < self.distance_to_goal(self.g_best_x, self.g_best_y)):
                    self.g_best_x, self.g_best_y = particle.x, particle.y
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
width, height = 9, 9
grid = Grid(width, height)
start_x, start_y = 0, 0
goal_x, goal_y = 8, 8
grid.generate_maze(start_x, start_y, goal_x, goal_y)

# Create agents
agents = [
    #Agent(0, 0, AStar(goal_x, goal_y)),
    #Agent(0, 0, ThetaStar(goal_x, goal_y)),
    #Agent(0, 0, JPS(goal_x, goal_y)),
    #Agent(0, 0, DFS(goal_x, goal_y)),
    Agent(0, 0, DStarLite(goal_x, goal_y)),
    #Agent(0, 0, PSO(goal_x, goal_y)),
]

# Run the simulation
simulation = Simulation(grid, agents, goal_x, goal_y)
simulation.run()


# import random
# import time

# # create 10 random grids of different sizes
# grids = []
# for i in range(1):
#     size = random.randint(900, 1000)
#     grids.append(Grid(size, size))

# # create the AStar, ThetaStar and JPS algorithms
# astar = AStar(goal_x, goal_y)
# thetastar = ThetaStar(goal_x, goal_y)
# jps = JPS(goal_x, goal_y)
# dfs = DFS(goal_x, goal_y)
# dstarlite = DStarLite(goal_x, goal_y)

# # measure the time it takes for each algorithm to find a path on each grid
# astar_times = []
# thetastar_times = []
# jps_times = []
# dfs_times = []
# dstarlite_times = []
# for grid in grids:
#     # set the starting position and goal position for each grid
#     start_x, start_y = random.randint(0, grid.width - 1), random.randint(0, grid.height - 1)
#     goal_x, goal_y = random.randint(0, grid.width - 1), random.randint(0, grid.height - 1)
#     grid.generate_maze(start_x, start_y, goal_x, goal_y)

#     # update AStar and ThetaStar goal positions
#     astar.goal_x, astar.goal_y = goal_x, goal_y
#     thetastar.goal_x, thetastar.goal_y = goal_x, goal_y
#     jps.goal_x, jps.goal_y = goal_x, goal_y
#     dfs.goal_x, dfs.goal_y = goal_x, goal_y
#     dstarlite.goal_x, dstarlite.goal_y = goal_x, goal_y

#     # measure AStar time
#     start_time = time.time()
#     path = astar.find_path((start_x, start_y), grid)
#     end_time = time.time()
#     astar_time = end_time - start_time
#     astar_times.append(astar_time)

#     # measure ThetaStar time
#     start_time = time.time()
#     path = thetastar.find_path((start_x, start_y), grid)
#     end_time = time.time()
#     thetastar_time = end_time - start_time
#     thetastar_times.append(thetastar_time)
    
#     # measure JPS time
#     start_time = time.time()
#     path = jps.find_path((start_x, start_y), grid)
#     end_time = time.time()
#     jps_time = end_time - start_time
#     jps_times.append(jps_time)
    
#     # measure DFS time
#     start_time = time.time()
#     path = dfs.find_path((start_x, start_y), grid)
#     end_time = time.time()
#     dfs_time = end_time - start_time
#     dfs_times.append(dfs_time)
    
#     # measure DStarLite time
#     start_time = time.time()
#     path = dstarlite.find_path((start_x, start_y), grid)
#     end_time = time.time()
#     dstarlite_time = end_time - start_time
#     dstarlite_times.append(dstarlite_time)

# # print the average time for each algorithm
# print(f"AStar average time: {sum(astar_times)/len(astar_times):.6f} seconds")
# print(f"ThetaStar average time: {sum(thetastar_times)/len(thetastar_times):.6f} seconds")
# print(f"JPS average time: {sum(jps_times)/len(jps_times):.6f} seconds")
# print(f"DFS average time: {sum(dfs_times)/len(dfs_times):.6f} seconds")
# print(f"DStarLite average time: {sum(dstarlite_times)/len(dstarlite_times):.6f} seconds")


"""
100 random grids of different sizes from 50x50 to 100x100
AStar average time: 0.017849 seconds
ThetaStar average time: 0.029060 seconds
JPS average time: 0.012722 seconds
DFS average time: 0.011744 seconds
DStarLite average time: 0.050464 seconds
"""
"""
1 random grids of different sizes from 900x900 to 1000x1000
AStar average time: 26.007051 seconds
ThetaStar average time: 28.495731 seconds
JPS average time: 2.713599 seconds
DFS average time: 224.046146 seconds
DStarLite average time: 15.427422 seconds
"""

# http://www.codenamepandey.com/movementalgo
# https://en.wikipedia.org/wiki/Multi-agent_pathfinding
# 1 change to grid.get_cost of that cell and neighbor
# https://gamedev.stackexchange.com/questions/141688/how-to-optimize-pathfinding-on-a-very-large-dynamic-2d-grid
# https://grail.cs.washington.edu/projects/crowd-flows/78-treuille.pdf
# PSO also draw all particles

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

6. 雙向搜索算法：
   優點：可以減少搜索的節點數量，從而提高搜索速度。
   缺點：需要知道起點和終點，不能用於單源最短路徑問題。
    適用情況：需要求解特定兩點間的最短路徑問題。
"""

"""
The DARPA Grand Challenge was a series of autonomous vehicle races organized by the Defense Advanced Research Projects Agency (DARPA) in the early 2000s. The goal of the Grand Challenge was to accelerate the development of autonomous vehicle technology for military use.

The first DARPA Grand Challenge was held in 2004 and involved a 142-mile course through the Mojave Desert in California. The challenge was to build a fully autonomous vehicle that could navigate the course without any human intervention. None of the teams were able to complete the course, with the farthest any vehicle made it being 7.4 miles.

The second DARPA Grand Challenge was held in 2005 and had a more difficult course that covered 132 miles through the desert. This time, five teams were able to complete the course, with the winner being "Stanley," a self-driving car developed by a team from Stanford University.

The success of the DARPA Grand Challenge helped to kickstart the development of autonomous vehicle technology and led to further advancements in the field. Today, autonomous vehicles are becoming more common in both military and civilian applications, and the technology continues to evolve rapidly.
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
https://zhuanlan.zhihu.com/p/349074802
"""
