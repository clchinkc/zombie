
import heapq
import random
from enum import Enum

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
        self.grid = np.full((height, width), CellType.EMPTY)

    def set_obstacle(self, x, y):
        self.grid[y][x] = CellType.OBSTACLE

    def set_goal(self, x, y):
        self.grid[y][x] = CellType.GOAL

    def is_valid_move(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] != CellType.OBSTACLE

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

    def heuristic(self, x, y):
        return abs(x - self.goal_x) + abs(y - self.goal_y)

    def move(self, x, y, grid):
        open_list = []
        heapq.heappush(open_list, (0, (x, y)))
        came_from = dict()
        g_score = {pos: float("inf") for pos in np.ndindex(grid.height, grid.width)}
        g_score[(x, y)] = 0

        while open_list:
            _, current = heapq.heappop(open_list)
            current_y, current_x = current

            if (current_x, current_y) == (self.goal_x, self.goal_y):
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[-2]

            neighbors = [(current_y + dy, current_x + dx) for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
            for neighbor_y, neighbor_x in neighbors:
                if not grid.is_valid_move(neighbor_y, neighbor_x):
                    continue

                tentative_g_score = g_score[(current_y, current_x)] + 1
                if tentative_g_score < g_score[(neighbor_y, neighbor_x)]:
                    came_from[(neighbor_y, neighbor_x)] = (current_y, current_x)
                    g_score[(neighbor_y, neighbor_x)] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor_y, neighbor_x)
                    heapq.heappush(open_list, (f_score, (neighbor_y, neighbor_x)))

        return x, y

class Swarm:
    def __init__(self, goal_x, goal_y):
        self.num_particles = 1
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
width, height = 10, 10
grid = Grid(width, height)
grid.set_obstacle(3, 3)
grid.set_obstacle(4, 4)
goal_x, goal_y = 9, 9
grid.set_goal(goal_x, goal_y)

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
    # Agent(0, 0, RandomWalk()),
    Agent(0, 1, AStar(goal_x, goal_y)),
    Agent(0, 2, Swarm(goal_x, goal_y)),
]

# Run the simulation
run_simulation(grid, agents, goal_x, goal_y)

