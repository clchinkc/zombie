


"""
A simulation of a chase problem:

Define the environment: First, you need to define the environment in which the chase will take place. For example, let's say the environment is a two-dimensional grid, with obstacles (walls) randomly placed throughout.

Create the characters: Next, you need to create the characters involved in the chase. For example, you could have a "runner" character and a "chaser" character. The runner is trying to avoid the chaser, and the chaser is trying to catch the runner.

Implement movement: You need to implement movement for the characters. For example, the runner might move randomly around the grid, while the chaser moves towards the runner. You'll need to define the rules for movement, such as how far each character can move in a single turn.

Implement obstacles: You'll need to implement obstacles (walls) in the environment to make the chase more challenging. The runner will need to navigate around them while trying to avoid the chaser.

Implement catching: You need to define the rules for catching. For example, the chaser might catch the runner if they move onto the same grid cell.

Implement a scoring system: Finally, you need to implement a scoring system to determine the outcome of the chase. For example, if the runner is caught, the chaser wins. If the runner manages to avoid the chaser for a certain amount of time, they win.

Run the simulation: Once you've implemented all of the above, you can run the simulation and observe the results. You might want to run the simulation multiple times with different starting positions and obstacles to see how the outcome changes.
"""

import random

import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras import Model, layers

# Define constants
GRID_SIZE = 32
NUM_OBSTACLES = 10
NUM_EPISODES = 10
NUM_STEPS = 1000
MOVE_DELAY = 1000

# Initialize Pygame only for visualization (remove if you don't want to visualize)
import pygame

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class DQN(Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = layers.Conv2D(16, (3, 3), activation='relu')
        self.conv2 = layers.Conv2D(32, (3, 3), activation='relu')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(num_actions)

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class RunnerDQN:
    def __init__(self, num_actions, learning_rate=1e-3):
        self.model = DQN(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def train_step(self, state, action, reward, next_state, done):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        state = tf.expand_dims(state, -1)
        next_state = tf.expand_dims(tf.convert_to_tensor(next_state), 0)
        next_state = tf.expand_dims(next_state, -1)

        with tf.GradientTape() as tape:
            q_values = self.model(state)
            one_hot_action = tf.one_hot([action], num_actions)
            q_value = tf.reduce_sum(one_hot_action * q_values, axis=1)

            next_q_values = self.model(next_state)
            max_next_q_value = tf.reduce_max(next_q_values)

            target_q_value = tf.stop_gradient(reward + (1 - done) * 0.99 * max_next_q_value)
            loss = self.loss_function(tf.reshape(target_q_value, (-1,)), q_value)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(0, num_actions)
        else:
            state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            state = tf.expand_dims(state, -1)  # Add a channel dimension
            q_values = self.model(state)
            return np.argmax(q_values[0])

# Define the runner and chaser classes
class Runner:
    def __init__(self, x, y, num_actions):
        self.x = x
        self.y = y
        self.dqn = RunnerDQN(num_actions)

    def get_state(self):
        # Create a state representation based on the runner's position, obstacles' positions, and the chaser's position
        state = np.zeros((SCREEN_HEIGHT // GRID_SIZE, SCREEN_WIDTH // GRID_SIZE))
        state[self.y // GRID_SIZE, self.x // GRID_SIZE] = 1

        for obstacle in obstacles:
            state[obstacle.y // GRID_SIZE, obstacle.x // GRID_SIZE] = 2
            
        state[chaser.y // GRID_SIZE, chaser.x // GRID_SIZE] = 3

        return state

    def move(self, action):
        # Map the action index to a change in position
        action_mapping = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        dx, dy = action_mapping[action]

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

def play_episode(runner, chaser, obstacles, epsilon):
    runner.x = random.randint(0, SCREEN_WIDTH // GRID_SIZE - 1) * GRID_SIZE
    runner.y = random.randint(0, SCREEN_HEIGHT // GRID_SIZE - 1) * GRID_SIZE

    for i in range(NUM_OBSTACLES):
        x = random.randint(0, SCREEN_WIDTH // GRID_SIZE - 1) * GRID_SIZE
        y = random.randint(0, SCREEN_HEIGHT // GRID_SIZE - 1) * GRID_SIZE
        obstacles[i].x = x
        obstacles[i].y = y

    chaser.x = random.randint(0, SCREEN_WIDTH // GRID_SIZE - 1) * GRID_SIZE
    chaser.y = random.randint(0, SCREEN_HEIGHT // GRID_SIZE - 1) * GRID_SIZE

    for step in range(NUM_STEPS):
        state = runner.get_state()
        action = runner.dqn.get_action(state, epsilon)
        runner.move(action)
        chaser.move(runner.x, runner.y)

        next_state = runner.get_state()
        done = False
        reward = 1

        if runner.x == chaser.x and runner.y == chaser.y:
            done = True
            reward = -1000

        runner.dqn.train_step(state, action, reward, next_state, done)

        if done:
            break

    return step

# Create the runner, chaser, and obstacles
num_actions = 4
runner = Runner(0, 0, num_actions)
chaser = Chaser(SCREEN_WIDTH - GRID_SIZE, SCREEN_HEIGHT - GRID_SIZE)

obstacles = []
for i in range(NUM_OBSTACLES):
    x = random.randint(0, SCREEN_WIDTH // GRID_SIZE - 1) * GRID_SIZE
    y = random.randint(0, SCREEN_HEIGHT // GRID_SIZE - 1) * GRID_SIZE
    obstacles.append(Obstacle(x, y))

# Train the DQN
epsilon = 1.0
epsilon_decay = 0.995
for episode in range(NUM_EPISODES):
    steps = play_episode(runner, chaser, obstacles, epsilon)
    epsilon = max(epsilon * epsilon_decay, 0.1)
    print("Episode", episode, "completed in", steps, "steps.")

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# Main game loop
for i in range(10):
    running = True
    move_timer = 0
    epsilon = 1.0
    epsilon_decay = 0.995
    state = runner.get_state()
    action = runner.dqn.get_action(state, epsilon)
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move the runner and chaser
        move_timer += clock.tick()
        if move_timer >= MOVE_DELAY:
            # Runner's move
            state = runner.get_state()
            action = runner.dqn.get_action(state, epsilon)
            runner.move(action)
            next_state = runner.get_state()

            # Chaser's move
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
        done = False
        if runner.x == chaser.x and runner.y == chaser.y:
            print("Runner caught!")
            running = False
            done = True

    # Reset the runner, obstacles, and chaser positions randomly when the runner is caught
    runner.x = random.randint(0, SCREEN_WIDTH // GRID_SIZE - 1) * GRID_SIZE
    runner.y = random.randint(0, SCREEN_HEIGHT // GRID_SIZE - 1) * GRID_SIZE
    for i in range(NUM_OBSTACLES):
        x = random.randint(0, SCREEN_WIDTH // GRID_SIZE - 1) * GRID_SIZE
        y = random.randint(0, SCREEN_HEIGHT // GRID_SIZE - 1) * GRID_SIZE
        obstacles[i].x = x
        obstacles[i].y = y
    chaser.x = SCREEN_WIDTH - GRID_SIZE
    chaser.y = SCREEN_HEIGHT - GRID_SIZE

# Quit Pygame
pygame.quit()




