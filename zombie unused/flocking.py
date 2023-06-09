
from collections import deque
import pygame
import numpy as np
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
        self.position += self.velocity
        self.history.append(self.position.copy())  # Save position for drawing trails

        # Boundary conditions
        self.position[0] = max(0, min(self.position[0], width - 1))
        self.position[1] = max(0, min(self.position[1], height - 1))

        # Adjust velocity if hitting the boundaries
        self.velocity[0] = abs(self.velocity[0]) if self.position[0] == 0 or self.position[0] == width - 1 else self.velocity[0]
        self.velocity[1] = abs(self.velocity[1]) if self.position[1] == 0 or self.position[1] == height - 1 else self.velocity[1]

    def draw(self):
        # Draw the agent's trail
        for i, pos in enumerate(reversed(self.history)):
            color = tuple([int(c * (1 - i / len(self.history))) for c in self.color])
            pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), 3)


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

        # Limiting the maximum velocity to prevent explosion
        desired_velocity *= 0.5 if np.linalg.norm(desired_velocity) > 1 else 1 

        self.acceleration = 0.05 * (desired_velocity - self.velocity)  # The rate of acceleration change

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

        # Limiting the maximum velocity to prevent explosion
        desired_velocity *= 0.5 if np.linalg.norm(desired_velocity) > 1 else 1 

        self.acceleration = 0.05 * (desired_velocity - self.velocity)  # The rate of acceleration change

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

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Separate humans and zombies
    humans = [agent for agent in agents if isinstance(agent, Human)]
    zombies = [agent for agent in agents if isinstance(agent, Zombie)]

    # Fit the NearestNeighbors model to the agents' positions
    positions = np.array([agent.position for agent in humans])
    velocities = np.array([agent.velocity for agent in humans])
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(positions)

    # Update agent states
    for human in humans:
        distances, indices = nbrs.kneighbors(human.position.reshape(1, -1))

        # Get positions and velocities of the neighbors
        neighbors_positions = positions[indices[0]]
        neighbors_velocities = velocities[indices[0]]

        human.update_velocity(neighbors_positions, neighbors_velocities, [zombie.position for zombie in zombies])
        human.move()

        # Check if human has been turned into a zombie
        if any(np.linalg.norm(human.position - zombie.position) < 10.0 for zombie in zombies):
            agents.remove(human)
            agents.append(Zombie(human.position[0], human.position[1]))

    for zombie in zombies:
        zombie.update_velocity(humans)
        zombie.move()

    # Draw all agents
    screen.fill((0, 0, 0))
    for agent in agents:
        agent.draw()
    pygame.display.flip()

pygame.quit()



