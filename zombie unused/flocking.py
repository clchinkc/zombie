
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
from abc import ABC, abstractmethod
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
ZOMBIE_INFECTION_RADIUS = 10
ZOMBIE_ATTRACTION_WEIGHT = 1
HUMAN_DETECTION_RADIUS = 50
HUMAN_SEPARATION_RADIUS = 20
HUMAN_AVOIDANCE_RADIUS = 50
HUMAN_AVOIDANCE_WEIGHT = 100

SEPARATION_WEIGHT = 0.01
ALIGNMENT_WEIGHT = 0.125
COHESION_WEIGHT = 0.01

# Agent Constants
AGENT_RADIUS = 5
AGENT_SPEED = 1


class Agent(ABC):
    # Constructor
    def __init__(self, x, y, color):
        self.position = Vector2(x, y)
        self.velocity = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * AGENT_SPEED
        self.acceleration = Vector2(0, 0)
        self.color = color
        self.history = deque(maxlen=50)
        
    # Basic Movement Methods
    def move(self):
        self.update_velocity()
        self.update_position()
        self.bounce_off_boundaries()
        self.reset_acceleration()
        
    def update_velocity(self):
        self.velocity += self.acceleration
        self.velocity.scale_to_length(min(self.velocity.length(), AGENT_SPEED))
    
    def update_position(self):
        self.position += self.velocity
        self.history.append(self.position.copy())
    
    def bounce_off_boundaries(self):
        self.position.x = np.clip(self.position.x, 0, WIDTH)
        self.position.y = np.clip(self.position.y, 0, HEIGHT)
        
        if self.position.x == 0 or self.position.x == WIDTH:
            self.velocity.x = -self.velocity.x
        if self.position.y == 0 or self.position.y == HEIGHT:
            self.velocity.y = -self.velocity.y
    
    def reset_acceleration(self):
        self.acceleration = Vector2(0, 0)

    # Utility Methods
    def get_separated_agent_lists(self, agents):
        humans = [a for a in agents if isinstance(a, Human)]
        zombies = [a for a in agents if isinstance(a, Zombie)]
        return humans, zombies

    # General Flocking Behaviors
    @abstractmethod
    def calculate_forces(self, agents):
        separation = self.calculate_separation(agents) * SEPARATION_WEIGHT
        alignment = self.calculate_alignment(agents) * ALIGNMENT_WEIGHT
        cohesion = self.calculate_cohesion(agents) * COHESION_WEIGHT
        return separation + alignment + cohesion
    
    def calculate_separation(self, agents):

        steering = Vector2(0, 0)
        total_influence = 0
        epsilon = 1e-10  # small value to prevent division by zero

        for agent in agents:
            if agent is not self and isinstance(agent, type(self)):
                distance = self.position.distance_to(agent.position)
                if distance < HUMAN_SEPARATION_RADIUS:
                    diff = self.position - agent.position
                    influence = 1 / (distance + epsilon)
                    steering += diff * influence
                    total_influence += influence

        if total_influence:
            steering /= total_influence
        return steering

    def calculate_alignment(self, agents):
        average_velocity = Vector2(0, 0)
        total = 0
        for agent in agents:
            if agent is not self and isinstance(agent, type(self)):
                average_velocity += agent.velocity
                total += 1
        if total:
            average_velocity /= total
        return average_velocity - self.velocity

    def calculate_cohesion(self, agents):
        center_of_mass = Vector2(self.position.x, self.position.y)
        total = 0
        for agent in agents:
            if agent is not self and isinstance(agent, type(self)):
                center_of_mass += agent.position
                total += 1
        if total:
            center_of_mass /= total
        return center_of_mass - self.position

    def add_friction_and_noise(self, force):
        FRICTION_COEFF = -0.01
        NOISE_RANGE = 0.02
        friction = FRICTION_COEFF * force
        noise = Vector2(random.uniform(-NOISE_RANGE, NOISE_RANGE), random.uniform(-NOISE_RANGE, NOISE_RANGE))
        return friction + noise

class Human(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, HUMAN_COLOR)

    # Overridden Method for Specific Behavior
    def calculate_forces(self, agents):
        humans, zombies = self.get_separated_agent_lists(agents)
        
        separation = self.calculate_separation(humans) * SEPARATION_WEIGHT
        alignment = self.calculate_alignment(humans) * ALIGNMENT_WEIGHT
        cohesion = self.calculate_cohesion(humans) * COHESION_WEIGHT
        avoidance = self.calculate_avoidance(zombies) * HUMAN_AVOIDANCE_WEIGHT
        
        combined_force = separation + alignment + cohesion + avoidance
        return combined_force + self.add_friction_and_noise(combined_force)

    # Specialized Method
    def calculate_avoidance(self, agents):
        steering = Vector2(0, 0)
        for agent in agents:
            distance = self.position.distance_to(agent.position)
            if distance < HUMAN_AVOIDANCE_RADIUS:  # Humans avoid zombies
                diff = self.position - agent.position
                steering += diff
        return steering

class Zombie(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, ZOMBIE_COLOR)

    # Overridden Method for Specific Behavior
    def calculate_forces(self, agents):
        humans, _ = self.get_separated_agent_lists(agents)
        
        attraction = self.calculate_attraction(humans)
        
        return attraction + self.add_friction_and_noise(attraction)

    # Specialized Method
    def calculate_attraction(self, agents):
        closest_human = min(agents, key=lambda human: self.position.distance_to(human.position), default=Human(self.position.x, self.position.y))
        attraction = (closest_human.position - self.position) * ZOMBIE_ATTRACTION_WEIGHT
        return attraction


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

    # 1. Initialization and setup methods

    def __init__(self, num_humans=80, num_zombies=20):
        pygame.init()
        self.setup_screen()
        self.create_agents(num_humans, num_zombies)
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False

    def setup_screen(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

    def create_agents(self, num_humans, num_zombies):
        self.agents = []
        self.humans = [Human(*self.random_position()) for _ in range(num_humans)]
        self.zombies = [Zombie(*self.random_position()) for _ in range(num_zombies)]
        self.agents.extend(self.humans)
        self.agents.extend(self.zombies)
        self.agent_sprites = Group([AgentSprite(agent) for agent in self.agents])
        self.trail_sprites = Group([TrailSprite(agent) for agent in self.agents])
        
    def random_position(self):
        position = Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        while any(position.distance_to(agent.position) < 2 * AGENT_RADIUS for agent in self.agents):
            position = Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        return position

    # 2. Event handling methods

    def run(self):
        while self.running:
            self.handle_events()
            if not self.paused:
                self.update_agents()
                self.render_agents()
        pygame.quit()
        # sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_key_event(event.key)

    def handle_key_event(self, key):
        key_actions = {
            pygame.K_ESCAPE: self.stop_running,
            pygame.K_SPACE: self.toggle_pause,
            pygame.K_r: self.__init__,
        }
        action = key_actions.get(key)
        if action:
            action()

    def stop_running(self):
        self.running = False

    def toggle_pause(self):
        self.paused = not self.paused

    # 3. Agent update methods

    def update_agents(self):
        positions = np.array([agent.position for agent in self.agents])
        nbrs = NearestNeighbors(n_neighbors=10).fit(positions)

        self.update_human_agents(nbrs)
        self.update_zombie_agents(nbrs)

    def update_human_agents(self, nbrs):
        for human in list(self.humans):  # Using list to avoid RuntimeError due to list modification
            indices = nbrs.radius_neighbors(np.array([human.position]).reshape(1, -1), radius=HUMAN_DETECTION_RADIUS, return_distance=False)[0]
            neighbours = [self.agents[i] for i in indices]
            forces = human.calculate_forces(neighbours)
            human.acceleration += forces
            
            human.move()

            infected = any([human.position.distance_to(zombie.position) < ZOMBIE_INFECTION_RADIUS for zombie in neighbours if isinstance(zombie, Zombie)])
            if infected:
                self.convert_human_to_zombie(human)

    def convert_human_to_zombie(self, human):
        self.humans.remove(human)
        new_zombie = Zombie(human.position.x, human.position.y)
        self.zombies.append(new_zombie)
        self.agents.append(new_zombie)
        for group in [self.agent_sprites, self.trail_sprites]:
            group.remove([s for s in group if s.agent == human][0])
            group.add(AgentSprite(new_zombie) if group == self.agent_sprites else TrailSprite(new_zombie))

    def update_zombie_agents(self, nbrs):
        for zombie in self.zombies:
            indices = nbrs.radius_neighbors(np.array([zombie.position]).reshape(1, -1), radius=ZOMBIE_DETECTION_RADIUS, return_distance=False)[0]
            neighbours = [self.agents[i] for i in indices]
            forces = zombie.calculate_forces(neighbours)
            zombie.acceleration += forces
            zombie.move()

    # 4. Render and display methods

    def render_agents(self):
        self.screen.fill(BACKGROUND_COLOR)
        for group in [self.trail_sprites, self.agent_sprites]:
            group.update()
            for sprite in group:
                self.screen.blit(sprite.image, sprite.rect)
        self.display_on_screen_info()
        pygame.display.flip()
        self.clock.tick(120)

    def display_on_screen_info(self):
        font = pygame.font.SysFont('Arial', 20)
        self.screen.blit(font.render(f'Humans: {len(self.humans)}', False, WORD_COLOR), (10, 10))
        self.screen.blit(font.render(f'Zombies: {len(self.zombies)}', False, WORD_COLOR), (10, 30))


if __name__ == '__main__':
    Simulation().run()



"""
Here are a few suggestions for optimizing this code:

Use pygame.sprite.collide_circle() to check for collisions between sprites.

Use pygame.sprite.spritecollide() to check for collisions between sprites and a group of sprites.

Use pygame.sprite.groupcollide() to check for collisions between humans and zombies more efficiently

Highlight agents not in sync with the flock (moving with large forces) by changing their color.

Window size can be number of cell in each side of the grid times size of each cell in the grid.

Please update the code according to these comments on pygame optimization.
"""

"""
Future improvements:
Introduce a vision range for Pursuer can only "see" Evader within a certain vision_range and move towards them.
If a Pursuer is close enough to a Evader, the Pursuer "kills" the Evader.
Introduce obstacles in the game environment that the pursuers and evader must navigate around and adjust the behaviour of both Pursuer and Evader to plan their movements. The fitness function would also need to be adjusted accordingly.
Introduce pathfinding so that zombies and humans can navigate around obstacles.
Make humans smarter, maybe group together or run away more strategically.
Introduce a safe zone for humans. If humans reach this zone, they're safe. If there is no resources, then humans need to walk out of it.
Maybe introduce a time factor. Humans win if they avoid zombies for a certain period. Zombies win if they infect all humans before time runs out.
"""

"""
**Artificial Swarm Intelligence (ASI) in a Zombie Apocalypse Simulation**

Artificial Swarm Intelligence (ASI) is a concept inspired by swarm behaviors observed in animals, like birds flocking and ants working together. In a zombie apocalypse simulation, ASI algorithms can model and manage the behavior of survivors, zombies, and other elements. Implementing this in Python involves creating specific algorithms to mimic these behaviors. Below is a comprehensive overview:

**1. Survivor Group Dynamics and Movement:**
- ASI can simulate the movement and decision-making of survivor groups, mimicking emergent behaviors seen in animals like flocks of birds.
- Implement movement algorithms using techniques like random walk and flocking behavior.
- Model how survivors collectively decide on resource allocation, defense strategies, and movement patterns.

**2. Zombie Behavior Modeling and Interaction:**
- ASI can realistically simulate zombies' behavior based on swarm intelligence.
- Zombies might form hordes, coordinate attacks, and adapt to changes.
- Define interactions between zombies and survivors, considering factors like how zombies target survivors.

**3. Resource Distribution and Management:**
- ASI algorithms can mimic ants' task allocation based on pheromone trails to simulate resource distribution.
- Model the consumption and distribution of resources among survivor groups, ensuring efficient allocation.
- Simulate how survivors gather essentials like food and water while evading threats.

**4. Communication, Coordination, and Decision-Making:**
- ASI can model communication networks among survivors.
- Simulate information sharing and communication using messaging or signaling mechanisms.
- Incorporate swarm intelligence principles to model how leaders weigh group feedback to make informed choices.

**5. Adaptive Strategies and Learning:**
- Allow survivors to adapt strategies based on new discoveries.
- Implement mechanisms for entities to change behaviors according to the evolving conditions in the simulation.
- Entities can learn from their experiences, such as survivors learning to avoid high-zombie activity areas.

**6. Scavenging, Exploration, and Environment Dynamics:**
- Guide survivor exploration with algorithms that mimic ants' exploration patterns.
- The environment should adapt due to factors like weather, terrain, and zombie spread, influencing survivor behaviors.
- Develop a dynamic environment that evolves and ensure entities can react accordingly.

**7. Collaborative Defense:**
- Model how survivors construct defensive structures, drawing inspiration from how termites build complex formations.

**8. Visualization and Scenario Testing:**
- Use visualization libraries like Pygame or Matplotlib to graphically display the simulation.
- Render entities and environmental changes visually.
- The ASI-driven simulation can also be a testing platform to develop and evaluate survival strategies.

**9. Code Structure and Documentation:**
- Organize the code modularly using classes and functions.
- Clearly document the purpose of various components, algorithms, and parameters.

It's imperative to remember that while ASI can enhance the realism and complexity of a simulation, its effectiveness depends on the quality of underlying models, data, and assumptions. Implementing ASI-driven simulations requires both a deep understanding of swarm intelligence principles and adept Python programming skills. Consider utilizing external libraries and tools to simplify the development process.
"""

# https://github.com/warownia1/PythonCollider
