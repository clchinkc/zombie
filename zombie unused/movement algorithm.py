

"""
There are many ways to implement more complex algorithms for controlling the movements of people and zombies in a simulation. Here are a few examples of more advanced approaches you could consider:

Pathfinding algorithms: You could use algorithms such as A* or Dijkstra's algorithm to allow people and zombies to intelligently navigate the school layout and avoid obstacles. This could be useful for simulating more realistic or strategic movements, such as people trying to find the best route to escape the school or zombies trying to track down nearby survivors.

Flocking algorithms: You could use algorithms such as Boids or Reynolds' steering behaviors to simulate the collective movements of groups of people or zombies. This could be used to model the behavior of crowds or hordes, and could help to create more realistic and dynamic simulations.

Machine learning algorithms: You could use machine learning techniques such as reinforcement learning or decision trees to train people and zombies to make more intelligent decisions about their movements. This could allow the simulation to adapt and improve over time, and could potentially enable the people and zombies to learn from their experiences and make more effective decisions in the future.
"""


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

# Create agents
agents = [
    Agent(0, 0, RandomWalk()),
    Agent(0, 0, Swarm(goal_x, goal_y)),
    Agent(0, 0, AStar(goal_x, goal_y)),
]

# Run the simulation
run_simulation(grid, agents, goal_x, goal_y)

