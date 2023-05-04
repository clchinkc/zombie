

"""
There are many ways to implement more complex algorithms for controlling the movements of people and zombies in a simulation. Here are a few examples of more advanced approaches you could consider:

Pathfinding algorithms: You could use algorithms such as A* or Dijkstra's algorithm to allow people and zombies to intelligently navigate the school layout and avoid obstacles. This could be useful for simulating more realistic or strategic movements, such as people trying to find the best route to escape the school or zombies trying to track down nearby survivors.

Flocking algorithms: You could use algorithms such as Boids or Reynolds' steering behaviors to simulate the collective movements of groups of people or zombies. This could be used to model the behavior of crowds or hordes, and could help to create more realistic and dynamic simulations.

Machine learning algorithms: You could use machine learning techniques such as reinforcement learning or decision trees to train people and zombies to make more intelligent decisions about their movements. This could allow the simulation to adapt and improve over time, and could potentially enable the people and zombies to learn from their experiences and make more effective decisions in the future.
"""


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

class Swarm:
    def __init__(self, goal_x, goal_y):
        self.num_particles = 100
        self.max_iter = 100
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.best_path = None

    def move(self, x, y, grid):
        if self.best_path is None:
            self.find_best_path(x, y, grid)
        if not self.best_path:
            return x, y
        return self.best_path.pop(0)

    def find_best_path(self, x, y, grid):
        # Initialize particles
        particles = [self.create_particle(x, y, grid) for _ in range(self.num_particles)]

        # Main loop
        particle_history = []
        for _ in range(self.max_iter):
            for particle in particles:
                # Choose direction
                dx, dy = self.choose_direction(particle, grid)

                # Update position
                particle['x'], particle['y'] = self.constrain_position(particle['x'] + dx, particle['y'] + dy, grid)

                # Update the personal best if needed
                if self.distance(particle['x'], particle['y'], self.goal_x, self.goal_y) < self.distance(particle['p_best_x'], particle['p_best_y'], self.goal_x, self.goal_y):
                    particle['p_best_x'] = particle['x']
                    particle['p_best_y'] = particle['y']

            # Store particle history
            particle_history.append([p.copy() for p in particles])

            # Update the global best
            particles.sort(key=lambda p: self.distance(p['x'], p['y'], self.goal_x, self.goal_y))
            g_best_x, g_best_y = particles[0]['x'], particles[0]['y']

            # Check if the global best is the goal
            if (g_best_x, g_best_y) == (self.goal_x, self.goal_y):
                break

        self.best_path = self.construct_path(particle_history)
        return self.best_path

    def step(self):
        if self.best_path is None:
            raise ValueError("You must find the best path first using find_best_path() method.")
        if not self.best_path:
            return None
        return self.best_path.pop(0)

    def construct_path(self, particle_history):
        path = []
        for particles in particle_history:
            particles.sort(key=lambda p: self.distance(p['x'], p['y'], self.goal_x, self.goal_y))
            path.append((particles[0]['x'], particles[0]['y']))
        return path

    def distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def constrain_position(self, x, y, grid):
        new_x, new_y = int(round(x)), int(round(y))
        if grid.is_valid_move(new_x, new_y):
            return new_x, new_y
        else:
            return x, y

    def create_particle(self, x, y, grid):
        return {'x': x, 'y': y, 'p_best_x': x, 'p_best_y': y}

    def choose_direction(self, particle, grid):
        w = 0.5  # inertia weight
        c1 = 1  # cognitive weight
        c2 = 1  # social weight

        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        move_scores = []

        for dx, dy in moves:
            new_x, new_y = self.constrain_position(particle['x'] + dx, particle['y'] + dy, grid)

            if grid.is_valid_move(new_x, new_y):
                inertia_score = w * self.distance(particle['x'], particle['y'], new_x, new_y)
                cognitive_score = c1 * self.distance(particle['p_best_x'], particle['p_best_y'], new_x, new_y)
                social_score = c2 * self.distance(self.goal_x, self.goal_y, new_x, new_y)

                total_score = inertia_score + cognitive_score + social_score
                move_scores.append(total_score)
            else:
                # Move is not valid, assign a high score to avoid it
                move_scores.append(float('inf'))

        # Choose the move with the lowest score
        best_move_index = move_scores.index(min(move_scores))
        return moves[best_move_index]

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

        open_list = PriorityQueue()
        open_list.put((0, (start_pos, [start_pos])))
        visited = set()

        g_scores = {start_pos: 0}
        f_scores = {start_pos: self.heuristic(*start_pos)}

        while not open_list.empty():
            _, (current, path) = open_list.get()
            if current == end_pos:
                return path

            neighbors = self.find_neighbors(grid, current)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                tentative_g_score = g_scores[current] + 1
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + self.heuristic(*neighbor)
                    open_list.put((f_scores[neighbor], (neighbor, path + [neighbor])))
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
        if not grid.is_valid_move(*start) or not grid.is_valid_move(*end):
            return False
        
        x0, y0 = start
        x1, y1 = end
        dy = y1 - y0
        dx = x1 - x0
        sy = 1 if dy >= 0 else -1
        sx = 1 if dx >= 0 else -1
        dy = abs(dy)
        dx = abs(dx)
        f = 0
        is_valid_move = grid.is_valid_move
        offset_x = (1 ^ sx) >> 1
        offset_y = (1 ^ sy) >> 1

        if dx > dy:
            while x0 != x1:
                x0 += sx
                f += dy
                if f >= dx:
                    y0 += sy
                    f -= dx
                    if not is_valid_move(x0 - offset_x, y0 - offset_y):
                        return False
                if not is_valid_move(x0 - offset_x, y0 - offset_y):
                    return False
        else:
            while y0 != y1:
                y0 += sy
                f += dx
                if f >= dy:
                    x0 += sx
                    f -= dy
                    if not is_valid_move(x0 - offset_x, y0 - offset_y):
                        return False
                if not is_valid_move(x0 - offset_x, y0 - offset_y):
                    return False

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

    def find_path(self, start_pos, grid):
        end_pos = (self.goal_x, self.goal_y)

        open_list = PriorityQueue()
        open_list.put((0, (start_pos, [start_pos])))
        visited = set()

        g_scores = {start_pos: 0}
        f_scores = {start_pos: self.heuristic(*start_pos)}
        came_from = {}

        while not open_list.empty():
            _, (current, path) = open_list.get()
            if current == end_pos:
                return path

            neighbors = self.find_neighbors(grid, current, came_from)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                if self.line_of_sight(grid, current, neighbor):
                    new_g_score = g_scores[current] + math.sqrt((neighbor[0] - current[0])**2 + (neighbor[1] - current[1])**2)
                else:
                    new_g_score = g_scores[current] + 1

                if neighbor not in g_scores or new_g_score < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = new_g_score
                    f_scores[neighbor] = new_g_score + self.heuristic(*neighbor)
                    open_list.put((f_scores[neighbor], (neighbor, path + [neighbor])))
                    visited.add(neighbor)
                    
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
        if not grid.is_valid_move(new_x, new_y):
            return None

        if (new_x, new_y) == (self.goal_x, self.goal_y):
            return (new_x, new_y)

        if dx != 0 and dy != 0:
            if (grid.is_valid_move(new_x - dx, new_y) and not grid.is_valid_move(x - dx, y)) or \
               (grid.is_valid_move(new_x, new_y - dy) and not grid.is_valid_move(x, y - dy)):
                return (new_x, new_y)
        else:
            if dx != 0:
                if (grid.is_valid_move(new_x, new_y - 1) and not grid.is_valid_move(x, y - 1)) or \
                   (grid.is_valid_move(new_x, new_y + 1) and not grid.is_valid_move(x, y + 1)):
                    return (new_x, new_y)
            else:
                if (grid.is_valid_move(new_x - 1, new_y) and not grid.is_valid_move(x - 1, y)) or \
                   (grid.is_valid_move(new_x + 1, new_y) and not grid.is_valid_move(x + 1, y)):
                    return (new_x, new_y)

        return self.jump(grid, (new_x, new_y), direction)

    def find_jump_points(self, grid, current):
        jump_points = []
        neighbors = self.find_neighbors(grid, current)

        for neighbor in neighbors:
            direction = (neighbor[0] - current[0], neighbor[1] - current[1])
            jump_point = self.jump(grid, current, direction)

            if jump_point is not None:
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
            if current == end_pos:
                return self.reconstruct_path(path)

            jump_points = self.find_jump_points(grid, current)
            for jump_point in jump_points:
                if jump_point in visited:
                    continue

                tentative_g_score = g_scores[current] + math.sqrt((jump_point[0] - current[0])**2 + (jump_point[1] - current[1])**2)
                if jump_point not in g_scores or tentative_g_score < g_scores[jump_point]:
                    g_scores[jump_point] = tentative_g_score
                    f_scores[jump_point] = tentative_g_score + self.heuristic(*jump_point)
                    open_list.put((f_scores[jump_point], (jump_point, path + [jump_point])))
                    visited.add(jump_point)
                    
        return []

    def move(self, x, y, grid):
        path = self.find_path((x, y), grid)
        if len(path) > 1:
            next_step = path[1]
            return next_step
        return x, y


def draw_grid(screen, grid):
    for y in range(grid.height):
        for x in range(grid.width):
            cell_color = WHITE
            if grid.grid[y][x] == CellType.OBSTACLE:
                cell_color = BLACK
            elif grid.grid[y][x] == CellType.GOAL:
                cell_color = YELLOW

            pygame.draw.rect(screen, cell_color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)
            pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

def draw_agents(screen, agents):
    for idx, agent in enumerate(agents):
        agent_color = GREEN if idx % 2 == 0 else BLUE
        pygame.draw.circle(screen, agent_color, (agent.x * CELL_SIZE + CELL_SIZE // 2, agent.y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2)

def run_simulation(grid, agents, goal_x, goal_y, max_iterations=1000):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Grid-Based Simulation")
    clock = pygame.time.Clock()

    iterations = 0
    agents_reached_goal = set()

    running = True
    while running and iterations < max_iterations and len(agents_reached_goal) < len(agents):
        screen.fill(WHITE)
        draw_grid(screen, grid)
        draw_agents(screen, agents)
        pygame.display.flip()

        for idx, agent in enumerate(agents):
            if idx in agents_reached_goal:
                continue

            agent.move(grid)

            if (agent.x, agent.y) == (goal_x, goal_y):
                agents_reached_goal.add(idx)
                print(f"Agent {idx} reached the goal at iteration {iterations}")

        iterations += 1
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

    if len(agents_reached_goal) == len(agents):
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


# Pygame constants
CELL_SIZE = 40
WINDOW_WIDTH = width * CELL_SIZE
WINDOW_HEIGHT = height * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
FPS = 10

# # Create agents
# agents = [
#     Agent(0, 0, RandomWalk()),
#     Agent(0, 0, Swarm(goal_x, goal_y)),
#     Agent(0, 0, AStar(goal_x, goal_y)),
#     Agent(0, 0, ThetaStar(goal_x, goal_y)),
#     Agent(0, 0, JPS(goal_x, goal_y)),
# ]

# # Run the simulation
# run_simulation(grid, agents, goal_x, goal_y)


import random
import time

# create 10 random grids of different sizes
grids = []
for i in range(100):
    size = random.randint(1000, 2000)
    grids.append(Grid(size, size))

# create the AStar, ThetaStar and JPS algorithms
astar = AStar(goal_x, goal_y)
thetastar = ThetaStar(goal_x, goal_y)
jps = JPS(goal_x, goal_y)

# measure the time it takes for each algorithm to find a path on each grid
astar_times = []
thetastar_times = []
jps_times = []
for grid in grids:
    # set the starting position and goal position for each grid
    start_x, start_y = random.randint(0, grid.width - 1), random.randint(0, grid.height - 1)
    goal_x, goal_y = random.randint(0, grid.width - 1), random.randint(0, grid.height - 1)

    # update AStar and ThetaStar goal positions
    astar.goal_x, astar.goal_y = goal_x, goal_y
    thetastar.goal_x, thetastar.goal_y = goal_x, goal_y
    jps.goal_x, jps.goal_y = goal_x, goal_y

    # measure AStar time
    start_time = time.time()
    path = astar.find_path((start_x, start_y), grid)
    end_time = time.time()
    astar_time = end_time - start_time
    astar_times.append(astar_time)

    # measure ThetaStar time
    start_time = time.time()
    path = thetastar.find_path((start_x, start_y), grid)
    end_time = time.time()
    thetastar_time = end_time - start_time
    thetastar_times.append(thetastar_time)
    
    # measure JPS time
    start_time = time.time()
    path = jps.find_path((start_x, start_y), grid)
    end_time = time.time()
    jps_time = end_time - start_time
    jps_times.append(jps_time)

# print the average time for each algorithm
print(f"AStar average time: {sum(astar_times) / len(astar_times):.6f} seconds")
print(f"ThetaStar average time: {sum(thetastar_times) / len(thetastar_times):.6f} seconds")
print(f"JPS average time: {sum(jps_times) / len(jps_times):.6f} seconds")



# http://www.codenamepandey.com/movementalgo


