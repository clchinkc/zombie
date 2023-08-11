
"""
Here are some steps you can follow to develop a simulation of flocking behavior using a multi-agent system:

1. **Agent Representation**

Each agent in the simulation should be represented as an entity with properties such as position, velocity, and acceleration. The position of an agent represents its location in the simulation environment. The velocity of an agent represents its speed and direction of movement. The acceleration of an agent represents the rate at which its velocity is changing.

2. **Neighboring Agents**

A mechanism must be defined to determine the neighboring agents for each individual agent. One way to do this is to use a proximity-based approach, such as a fixed radius around each agent. This means that an agent will consider all other agents that are within a certain distance of it to be its neighbors. Another way to determine neighboring agents is to use a k-nearest neighbors approach. This means that an agent will consider the k agents that are closest to it to be its neighbors.

3. **Interaction Rules**

Rules must be defined that govern how agents interact with each other. These rules should incorporate the principles of cohesion, alignment, and separation. The following are some examples of interaction rules:

* **Cohesion:** Agents should move towards the center of mass of their neighbors to maintain a cohesive group.
* **Alignment:** Agents should align their velocities with the average velocity of their neighbors to achieve a sense of directionality.
* **Separation:** Agents should maintain a minimum distance from each other to avoid collisions.

The relative weighting of these rules can be adjusted to control the behavior of the flock. For example, if the cohesion rule is given a higher weight than the alignment rule, the flock will tend to form a more cohesive group. If the alignment rule is given a higher weight than the cohesion rule, the flock will tend to move in a more coordinated manner.

4. **Environment Constraints**

The boundaries and obstacles within the simulation environment must be considered. Agents should respond to these constraints to avoid collisions and exhibit realistic flocking behavior. For example, if there is a wall at the edge of the simulation environment, agents should avoid colliding with the wall by changing their direction of movement.

5. **Visualization**

A visual representation of the simulation should be developed, where agents and their interactions are visually displayed. This can help analyze the emergent patterns and behavior of the flock. For example, a visualization of a flock of birds might show the birds moving in a coordinated manner, avoiding obstacles, and changing direction to avoid collisions.

6. **Experimentation and Analysis**

Experiments should be conducted by varying parameters such as the number of agents, initial positions, interaction rules, and environmental constraints. The impact of these variations on the resulting flocking behavior should be analyzed. For example, an experiment might involve varying the number of agents in a flock to see how the flock's behavior changes.

7. **Documentation**

Clear documentation should be provided explaining the implementation details, the reasoning behind your design choices, and the insights gained from the experiments. This documentation can be used to help others understand your work and to build upon your work in the future.

Implement in python.
"""

import random
import sys
from abc import abstractmethod
from collections import deque

import numpy as np
import pygame
from pygame.math import Vector2
from pygame.sprite import Group, Sprite
from sklearn.neighbors import NearestNeighbors

# Window and Game Constants
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (0, 0, 0)
WORD_COLOR = (255, 255, 255)
HUMAN_COLOR = (0, 0, 255) # Humans are blue
ZOMBIE_COLOR = (255, 0, 0) # Zombies are red

# Simulation Constants
ZOMBIE_DETECTION_RADIUS = 50
ZOMBIE_ATTRACTION_FACTOR = 0.125
HUMAN_DETECTION_RADIUS = 50
HUMAN_AVOIDANCE_RADIUS = 10
HUMAN_AVOIDANCE_FACTOR = 0.75

SEPARATION_FACTOR = 0.125
ALIGNMENT_FACTOR = 0.0125
COHESION_FACTOR = 0.0125

# Agent Constants
AGENT_RADIUS = 5
AGENT_SPEED = 1


class Agent:
    def __init__(self, x, y, color):
        self.position = Vector2(x, y)
        self.velocity = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * AGENT_SPEED
        self.acceleration = Vector2(0, 0)
        self.color = color
        self.history = deque(maxlen=50)  # Stores the past positions for drawing trails
        
    def move(self):
        self.velocity += self.acceleration
        if self.velocity.length() > AGENT_SPEED:
            self.velocity.scale_to_length(AGENT_SPEED) # Limit speed to AGENT_SPEED
        self.position += self.velocity
        
        # Boundary conditions (use min/max to avoid going off the screen)
        self.position.x = min(max(self.position.x, 0), WIDTH)
        self.position.y = min(max(self.position.y, 0), HEIGHT)
        
        # Adjust velocity if hitting the boundaries (stop and turn around)
        if self.position.x == 0 or self.position.x == WIDTH:
            self.velocity.x = -self.velocity.x
        if self.position.y == 0 or self.position.y == HEIGHT:
            self.velocity.y = -self.velocity.y
        
        self.history.append(self.position.copy()) # Store the current position in the history


    # General cohesion, alignment, and separation behaviors. To be overridden by specializations.
    @abstractmethod
    def calculate_forces(self, agents):
        separation = self.calculate_separation(agents) * SEPARATION_FACTOR
        alignment = self.calculate_alignment(agents) * ALIGNMENT_FACTOR
        cohesion = self.calculate_cohesion(agents) * COHESION_FACTOR
        return separation + alignment + cohesion
    
    def calculate_separation(self, agents):
        steering = Vector2(0, 0)
        count = 0
        for agent in agents:
            if agent is self or not isinstance(agent, type(self)):
                continue
            diff = self.position - agent.position
            diff /= self.position.distance_to(agent.position)
            steering += diff
            count += 1
        return steering if count == 0 else steering / count

    def calculate_alignment(self, agents):
        average_velocity = Vector2(0, 0)
        if len(agents) <= 1:
            return average_velocity
        for agent in agents:
            if agent is self:
                continue
            average_velocity += agent.velocity
        average_velocity /= len(agents) - 1
        return average_velocity - self.velocity

    def calculate_cohesion(self, agents):
        center_of_mass = Vector2(0, 0)
        if len(agents) <= 1:
            return center_of_mass
        for agent in agents:
            if agent is self:
                continue
            center_of_mass += agent.position
        center_of_mass /= len(agents) - 1
        return center_of_mass - self.position

class Human(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, HUMAN_COLOR)

    def calculate_forces(self, agents):
        humans = [a for a in agents if isinstance(a, Human)]
        zombies = [a for a in agents if isinstance(a, Zombie)]
        separation = self.calculate_separation(humans) * SEPARATION_FACTOR
        alignment = self.calculate_alignment(humans) * ALIGNMENT_FACTOR
        cohesion = self.calculate_cohesion(humans) * COHESION_FACTOR
        avoidance = self.calculate_avoidance(zombies) * HUMAN_AVOIDANCE_FACTOR
        return separation + alignment + cohesion + avoidance

    def calculate_avoidance(self, agents):
        steering = Vector2(0, 0)
        for agent in agents:
            distance = self.position.distance_to(agent.position)
            if distance < HUMAN_AVOIDANCE_RADIUS:  # Humans avoid zombies
                diff = self.position - agent.position
                diff /= distance ** 2 # Inverse square law for repulsion
                steering -= diff
        return steering

class Zombie(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, ZOMBIE_COLOR)

    def calculate_forces(self, agents):
        humans = [a for a in agents if isinstance(a, Human)]
        attraction = self.calculate_attraction(humans) * ZOMBIE_ATTRACTION_FACTOR
        return attraction

    def calculate_attraction(self, agents):
        steering = Vector2(0, 0)
        for agent in agents:
            diff = agent.position - self.position
            diff /= self.position.distance_to(agent.position)  # Attraction force grows with closeness
            steering += diff
        return steering


class AgentSprite(Sprite):
    def __init__(self, agent):
        super().__init__()
        self.image = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA).convert_alpha()
        self.rect = self.image.get_rect()
        self.agent = agent

    def update(self):
        self.agent.move()
        self.image.fill((0, 0, 0, 0))
        pygame.draw.circle(self.image, self.agent.color, (int(self.agent.position.x), int(self.agent.position.y)), AGENT_RADIUS)
        self.rect.topleft = (0, 0)

class TrailSprite(Sprite):
    def __init__(self, agent):
        super().__init__()
        self.image = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA).convert_alpha()
        self.rect = self.image.get_rect()
        self.agent = agent

    def update(self):
        self.image.fill((0, 0, 0, 0))
        for i, pos in enumerate(reversed(self.agent.history)):
            color = tuple([int(c * (1 - i / len(self.agent.history))) for c in self.agent.color])
            pygame.draw.circle(self.image, color + (100,), (int(pos.x), int(pos.y)), AGENT_RADIUS)
        self.rect.topleft = (0, 0)


class Simulation:
    def __init__(self, num_humans=80, num_zombies=20):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.agents = [Human(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(num_humans)]
        self.agents += [Zombie(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(num_zombies)]
        self.agent_sprites = Group(AgentSprite(agent) for agent in self.agents)
        self.trail_sprites = Group(TrailSprite(agent) for agent in self.agents)
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_key_event(event.key)

            if not self.paused:
                self.update_agents()
                self.render_agents()

        sys.exit()

    def handle_key_event(self, key):
        key_actions = {
            pygame.K_ESCAPE: lambda: setattr(self, 'running', False),
            pygame.K_SPACE: lambda: setattr(self, 'paused', not self.paused),
            pygame.K_r: lambda: self.__init__(),
        }
        if key in key_actions:
            key_actions[key]()

    def update_agents(self):
        # Separate humans and zombies
        humans = [agent for agent in self.agents if isinstance(agent, Human)]
        zombies = [agent for agent in self.agents if isinstance(agent, Zombie)]

        # Fit the NearestNeighbors model to the agents' positions
        positions = np.array([agent.position for agent in self.agents])
        nbrs = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(positions)

        # Update agent states
        for human in humans:
            # Update based on neighbours
            indices = nbrs.radius_neighbors(np.array([human.position]).reshape(1, -1), radius=HUMAN_DETECTION_RADIUS, return_distance=False)[0]
            neighbours = [self.agents[i] for i in indices]
            forces = human.calculate_forces(neighbours)
            human.acceleration = forces
            human.move()

        for zombie in zombies:
            # Update based on neighbours
            indices = nbrs.radius_neighbors(np.array([zombie.position]).reshape(1, -1), radius=ZOMBIE_DETECTION_RADIUS, return_distance=False)[0]
            neighbours = [self.agents[i] for i in indices]
            forces = zombie.calculate_forces(neighbours)
            zombie.acceleration = forces
            zombie.move()

    def render_agents(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.trail_sprites.update()
        self.agent_sprites.update()
        for sprite in self.trail_sprites:
            self.screen.blit(sprite.image, sprite.rect)
        for sprite in self.agent_sprites:
            self.screen.blit(sprite.image, sprite.rect)
        # Displaying on-screen information
        humans_count = len([agent for agent in self.agents if isinstance(agent, Human)])
        zombies_count = len([agent for agent in self.agents if isinstance(agent, Zombie)])
        self.screen.blit(pygame.font.SysFont('Arial', 20).render(f'Humans: {humans_count}', False, WORD_COLOR), (10, 10))
        self.screen.blit(pygame.font.SysFont('Arial', 20).render(f'Zombies: {zombies_count}', False, WORD_COLOR), (10, 30))
        pygame.display.flip()
        self.clock.tick(60)

if __name__ == '__main__':
    Simulation().run()



