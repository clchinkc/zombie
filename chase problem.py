
"""
Sure, here's an example of how you could design a simulation of a chase problem:

Define the environment: First, you need to define the environment in which the chase will take place. For example, let's say the environment is a two-dimensional grid, with obstacles (walls) randomly placed throughout.

Create the characters: Next, you need to create the characters involved in the chase. For example, you could have a "runner" character and a "chaser" character. The runner is trying to avoid the chaser, and the chaser is trying to catch the runner.

Implement movement: You need to implement movement for the characters. For example, the runner might move randomly around the grid, while the chaser moves towards the runner. You'll need to define the rules for movement, such as how far each character can move in a single turn.

Implement obstacles: You'll need to implement obstacles (walls) in the environment to make the chase more challenging. The runner will need to navigate around them while trying to avoid the chaser.

Implement catching: You need to define the rules for catching. For example, the chaser might catch the runner if they move onto the same grid cell.

Implement a scoring system: Finally, you need to implement a scoring system to determine the outcome of the chase. For example, if the runner is caught, the chaser wins. If the runner manages to avoid the chaser for a certain amount of time, they win.

Run the simulation: Once you've implemented all of the above, you can run the simulation and observe the results. You might want to run the simulation multiple times with different starting positions and obstacles to see how the outcome changes.
"""

import random

import pygame

# Define constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
GRID_SIZE = 32
MOVE_DELAY = 200
NUM_OBSTACLES = 10

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# Define the runner and chaser classes
class Runner:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self):
        # Move randomly in one of four directions
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.x += dx * GRID_SIZE
        self.y += dy * GRID_SIZE
        
        # Check bounds
        self.x = max(0, min(self.x, SCREEN_WIDTH - GRID_SIZE))
        self.y = max(0, min(self.y, SCREEN_HEIGHT - GRID_SIZE))
        
        # Check for collision with obstacles
        for obstacle in obstacles:
            if self.x == obstacle.x and self.y == obstacle.y:
                self.x -= dx * GRID_SIZE
                self.y -= dy * GRID_SIZE

    def draw(self):
        pygame.draw.rect(screen, BLUE, (self.x, self.y, GRID_SIZE, GRID_SIZE))

class Chaser:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, runner_x, runner_y):
        # Move towards the runner
        if self.x < runner_x:
            self.x += GRID_SIZE
        elif self.x > runner_x:
            self.x -= GRID_SIZE

        if self.y < runner_y:
            self.y += GRID_SIZE
        elif self.y > runner_y:
            self.y -= GRID_SIZE
            
        # Check bounds
        self.x = max(0, min(self.x, SCREEN_WIDTH - GRID_SIZE))
        self.y = max(0, min(self.y, SCREEN_HEIGHT - GRID_SIZE))
        
        # Check for collision with obstacles
        for obstacle in obstacles:
            if self.x == obstacle.x and self.y == obstacle.y:
                if self.x < runner_x:
                    self.x -= GRID_SIZE
                elif self.x > runner_x:
                    self.x += GRID_SIZE

                if self.y < runner_y:
                    self.y -= GRID_SIZE
                elif self.y > runner_y:
                    self.y += GRID_SIZE

    def draw(self):
        pygame.draw.rect(screen, RED, (self.x, self.y, GRID_SIZE, GRID_SIZE))

# Define the obstacle class
class Obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        pygame.draw.rect(screen, BLACK, (self.x, self.y, GRID_SIZE, GRID_SIZE))

# Create the runner, chaser, and obstacles
runner = Runner(0, 0)
chaser = Chaser(SCREEN_WIDTH - GRID_SIZE, SCREEN_HEIGHT - GRID_SIZE)
obstacles = []
for i in range(NUM_OBSTACLES):
    x = random.randint(0, SCREEN_WIDTH // GRID_SIZE - 1) * GRID_SIZE
    y = random.randint(0, SCREEN_HEIGHT // GRID_SIZE - 1) * GRID_SIZE
    obstacles.append(Obstacle(x, y))

# Main game loop
running = True
move_timer = 0
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move the runner and chaser
    move_timer += clock.tick()
    if move_timer >= MOVE_DELAY:
        runner.move()
        chaser.move(runner.x, runner.y)
        move_timer = 0

    # Draw the screen
    screen.fill(WHITE)
    for obstacle in obstacles:
        obstacle.draw()
    runner.draw()
    chaser.draw()
    pygame.display.flip()

    # Check for catching
    if runner.x == chaser.x and runner.y == chaser.y:
        print("Runner caught!")
        running = False

# Quit Pygame
pygame.quit()
