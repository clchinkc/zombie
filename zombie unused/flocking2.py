
import sys
from collections import deque

import numpy as np
import pygame
from pygame.sprite import Group, Sprite
from sklearn.neighbors import NearestNeighbors


class Agent:
    def __init__(self, x, y, color):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.01, 0.01])
        self.acceleration = np.array([0.0, 0.0])
        self.color = color
        self.history = deque(maxlen=50)  # Stores the past positions for drawing trails

    def move(self):
        self.velocity += self.acceleration
        self.velocity = np.clip(self.velocity, -1, 1)  # Limit maximum velocity to -1 and 1
        self.position += self.velocity
        self.history.append(self.position.copy())  # Save position for drawing trails

        # Boundary conditions
        self.position[0] = max(0, min(self.position[0], width - 1))
        self.position[1] = max(0, min(self.position[1], height - 1))

        # Adjust velocity if hitting the boundaries
        self.velocity[0] = abs(self.velocity[0]) if self.position[0] == 0 or self.position[0] == width - 1 else self.velocity[0]
        self.velocity[1] = abs(self.velocity[1]) if self.position[1] == 0 or self.position[1] == height - 1 else self.velocity[1]


class TrailSprite(Sprite):
    def __init__(self, agent):
        super().__init__()
        self.image = pygame.Surface((width, height), pygame.SRCALPHA).convert_alpha()
        self.rect = self.image.get_rect()
        self.agent = agent

    def update(self):
        # Clear the image
        self.image.fill((0, 0, 0, 0))
        # Draw the trail
        for i, pos in enumerate(reversed(self.agent.history)):
            color = tuple([int(c * (1 - i / len(self.agent.history))) for c in self.agent.color])
            pygame.draw.circle(self.image, color + (100,), (int(pos[0]), int(pos[1])), 5)
        self.rect.topleft = (0, 0)  # Always draw on the topleft and let pygame handle the blit optimization

class AgentSprite(Sprite):
    def __init__(self, agent):
        super().__init__()
        self.image = pygame.Surface((width, height), pygame.SRCALPHA).convert_alpha()
        self.rect = self.image.get_rect()
        self.agent = agent

    def update(self):
        self.agent.move()
        # Clear the image
        self.image.fill((0, 0, 0, 0))
        # Draw the agent
        pygame.draw.circle(self.image, self.agent.color, (int(self.agent.position[0]), int(self.agent.position[1])), 5)
        self.rect.topleft = (0, 0)  # Always draw on the topleft and let pygame handle the blit optimization


class Human(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, (0, 0, 255))  # Humans are blue

    def update_velocity(self, neighbors_positions, velocities, zombies_positions):
        center_of_mass = np.mean(neighbors_positions, axis=0)
        average_velocity = np.mean(velocities, axis=0)
        desired_velocity = np.array([0.0, 0.0])

        # Cohesion
        desired_velocity += 0.01 * (center_of_mass - self.position)

        # Alignment
        desired_velocity += 0.125 * (average_velocity - self.velocity)

        # Separation
        for position in neighbors_positions:
            if np.linalg.norm(self.position - position) < 20.0:
                desired_velocity -= (position - self.position)

        # Avoid zombies
        for position in zombies_positions:
            if np.linalg.norm(self.position - position) < 50.0:
                desired_velocity += (self.position - position)

        # Randomly move around
        friction = -0.01 * desired_velocity
        noise = 0.02 * np.random.normal(0, 1)
        desired_velocity += friction + noise

        # Limiting the maximum velocity by normalizing the vector
        if np.linalg.norm(desired_velocity) > 1:
            desired_velocity /= np.linalg.norm(desired_velocity)

        # The rate of acceleration change
        self.acceleration = 0.05 * (desired_velocity - self.velocity)

        # Limiting the maximum acceleration by normalizing the vector
        if np.linalg.norm(self.acceleration) > 1.0:
            self.acceleration /= np.linalg.norm(self.acceleration)


class Zombie(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, (255, 0, 0))  # Zombies are red

    def update_velocity(self, humans):
        desired_velocity = np.array([0.0, 0.0])

        # Move towards closest human
        closest_human = min(humans, key=lambda human: np.linalg.norm(self.position - human.position))
        desired_velocity += 0.01 * (closest_human.position - self.position)

        # Randomly move around
        friction = -0.01 * desired_velocity
        noise = 0.02 * np.random.normal(0, 1)
        desired_velocity += friction + noise

        # Limiting the maximum velocity by normalizing the vector
        if np.linalg.norm(desired_velocity) > 1:
            desired_velocity /= np.linalg.norm(desired_velocity)

        # The rate of acceleration change
        self.acceleration = 0.05 * (desired_velocity - self.velocity)

        # Limiting the maximum acceleration by normalizing the vector
        if np.linalg.norm(self.acceleration) > 1.0:
            self.acceleration /= np.linalg.norm(self.acceleration)

# Initialize pygame and create window
width, height = 800, 600
pygame.init()
screen = pygame.display.set_mode((width, height))

# Create agents
num_humans = 80
num_zombies = 20
agents = [Human(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(num_humans)]
agents += [Zombie(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(num_zombies)]

agent_sprites = Group(AgentSprite(agent) for agent in agents)
trail_sprites = Group(TrailSprite(agent) for agent in agents)

clock = pygame.time.Clock()

# Main loop
running = True # Variable to keep the main loop running
paused = False # Variable to handle pausing of the clock
key_actions = {
    pygame.K_ESCAPE: lambda: setattr(sys.modules[__name__], 'running', False),
    pygame.K_SPACE: lambda: setattr(sys.modules[__name__], 'paused', not paused)
}

while running:
    clock.tick(120)  # limit the frame rate to 60 FPS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in key_actions:
                key_actions[event.key]()

    if paused:
        continue  # Skip the rest of the loop if paused
    # Separate humans and zombies
    humans = [agent for agent in agents if isinstance(agent, Human)]
    zombies = [agent for agent in agents if isinstance(agent, Zombie)]

    # Fit the NearestNeighbors model to the agents' positions
    positions = np.array([agent.position for agent in humans])
    velocities = np.array([agent.velocity for agent in humans])
    nbrs = NearestNeighbors(radius=20.0, algorithm='ball_tree').fit(positions)

    # Update agent states
    for human in humans:
        indices = nbrs.kneighbors(human.position.reshape(1, -1), return_distance=False)

        # Get positions and velocities of the neighbors
        neighbors_positions = positions[indices[0]]
        neighbors_velocities = velocities[indices[0]]

        human.update_velocity(neighbors_positions, neighbors_velocities, [zombie.position for zombie in zombies])
        human.move()

        # Check if human has been turned into a zombie
        if any(np.linalg.norm(human.position - zombie.position) < 10.0 for zombie in zombies):
            agents.remove(human)
            new_zombie = Zombie(human.position[0], human.position[1])
            agents.append(new_zombie)
            agent_sprites.add(AgentSprite(new_zombie))
            trail_sprites.add(TrailSprite(new_zombie))

    for zombie in zombies:
        zombie.update_velocity(humans)
        zombie.move()

    # Draw all agents
    screen.fill((0, 0, 0))
    trail_sprites.update()
    agent_sprites.update()
    for sprite in trail_sprites:
        screen.blit(sprite.image, sprite.rect)
    for sprite in agent_sprites:
        screen.blit(sprite.image, sprite.rect)
    screen.blit(pygame.font.SysFont('Arial', 20).render('Humans: {}'.format(len(humans)), False, (255, 255, 255)), (10, 10))
    screen.blit(pygame.font.SysFont('Arial', 20).render('Zombies: {}'.format(len(zombies)), False, (255, 255, 255)), (10, 30))
    pygame.display.flip()

pygame.quit()


"""
The code you provided is a simulation of a system of "zombies" and "humans" using Python's pygame library for rendering and scikit-learn's NearestNeighbors for managing the agents' behaviors. The agents (both humans and zombies) move in a 2D space with certain rules, and the simulation is visualized using pygame. Here's a description of the major components:

Agents
There are two types of agents: humans and zombies, both inheriting from the Agent class.

Agent: Defines the basic attributes and methods that both human and zombie agents share. Agents have position, velocity, acceleration, color, and a history queue to keep track of the past positions for drawing trails.

Human: Represents a human agent. Humans interact with other nearby humans and avoid zombies. Their movement is determined by cohesion (staying near other humans), alignment (matching velocity with nearby humans), separation (avoiding getting too close to other humans), and avoidance of zombies.

Zombie: Represents a zombie agent. Zombies move towards the nearest human, trying to "turn" them into other zombies.

Sprites
There are two sprite classes for rendering:

TrailSprite: Used to draw trails behind agents, showing their paths.
AgentSprite: Renders the current positions of the agents (either human or zombie).
Simulation Setup
The window's width and height are defined, and a number of humans and zombies are created with random initial positions.

Main Loop
The main loop runs the simulation, including:

Event Handling: Manages user inputs such as exiting the game or pausing/unpausing the simulation.
Separation of Agents: Separates humans and zombies to process their behaviors differently.
Neighbors Finding: For each human, finds neighboring humans using NearestNeighbors.
State Update: Updates the state (position, velocity, etc.) of each agent based on the defined rules.
Zombie Transformation: Checks if any human has been turned into a zombie and updates the agents accordingly.
Rendering: Clears the screen and draws the trails and agents, and then updates the display.
HUD: Displays the current count of humans and zombies.

Concluding Remarks
The code provides an interesting and illustrative example of how simple rules can lead to complex behaviors. Humans try to stay together and avoid zombies, while zombies chase after humans. The combination of cohesion, separation, alignment, and avoidance rules creates emergent flocking behavior among the human agents, while the zombies give an external stimulus that affects this behavior.

This code can be further expanded or modified to create more complex simulations or games by adding additional behaviors, obstacles, or rules.
"""

# https://github.com/warownia1/PythonCollider

"""
Here are a few suggestions for optimizing this code:

Use pygame.sprite.collide_circle() to check for collisions between circles.

Use pygame.sprite.spritecollide() to check for collisions between a sprite and a group of sprites.

Use pygame.sprite.groupcollide() to check for collisions between two groups of sprites.

Please update the code according to these comments on pygame optimization.
"""
