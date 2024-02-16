

import random
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
HIGHLIGHT_COLOR = (255, 255, 0)  # Color for highlighting agents
LARGE_FORCE_THRESHOLD = 2

# Simulation Constants
ZOMBIE_DETECTION_RADIUS = 50
ZOMBIE_ATTRACTION_WEIGHT = 1
HUMAN_DETECTION_RADIUS = 50
HUMAN_SEPARATION_RADIUS = 20
HUMAN_AVOIDANCE_RADIUS = 50
HUMAN_AVOIDANCE_WEIGHT = 10
SEPARATION_WEIGHT = 0.1
ALIGNMENT_WEIGHT = 0.2
COHESION_WEIGHT = 0.01

# Agent Constants
AGENT_RADIUS = 10
AGENT_SPEED = 1


class Agent(ABC):
    # Constructor
    def __init__(self, x, y):
        self.position = Vector2(x, y)
        self.velocity = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * AGENT_SPEED
        self.acceleration = Vector2(0, 0)
        self.history = deque(maxlen=50)
        self.is_highlighted = False
        
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

    def is_force_large(self, force):
        return force.length() > LARGE_FORCE_THRESHOLD

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

    def calculate_friction_and_noise(self, force):
        FRICTION_COEFF = -0.01
        NOISE_RANGE = 0.02
        friction = FRICTION_COEFF * force
        noise = Vector2(random.uniform(-NOISE_RANGE, NOISE_RANGE), random.uniform(-NOISE_RANGE, NOISE_RANGE))
        return friction + noise

class Human(Agent):
    def __init__(self, x, y):
        super().__init__(x, y)

    # Overridden Method for Specific Behavior
    def calculate_forces(self, agents):
        humans, zombies = self.get_separated_agent_lists(agents)
        
        propulsion = self.calculate_propulsion()
        separation = self.calculate_separation(humans) * SEPARATION_WEIGHT
        alignment = self.calculate_alignment(humans) * ALIGNMENT_WEIGHT
        cohesion = self.calculate_cohesion(humans) * COHESION_WEIGHT
        avoidance = self.calculate_avoidance(zombies) * HUMAN_AVOIDANCE_WEIGHT
        friction_and_noise = self.calculate_friction_and_noise(self.velocity)
        combined_force = propulsion + separation + alignment + cohesion + avoidance + friction_and_noise
        return combined_force

    # Specialized Method
    def calculate_propulsion(self):
        # Humans move in a random direction based on their current normalized velocity
        return self.velocity.normalize()
    
    def calculate_avoidance(self, zombies):
        steering = Vector2(0, 0)
        for agent in zombies:
            distance = self.position.distance_to(agent.position)
            if distance < HUMAN_AVOIDANCE_RADIUS:  # Humans avoid zombies
                diff = self.position - agent.position
                steering += diff
        return steering

class Zombie(Agent):
    def __init__(self, x, y):
        super().__init__(x, y)

    # Overridden Method for Specific Behavior
    def calculate_forces(self, agents):
        humans, _ = self.get_separated_agent_lists(agents)
        
        propulsion = self.calculate_propulsion()
        attraction = self.calculate_attraction(humans)
        friction_and_noise = self.calculate_friction_and_noise(self.velocity)
        combined_force = propulsion + attraction + friction_and_noise
        return combined_force

    # Specialized Method
    def calculate_propulsion(self):
        # Humans move in a random direction based on their current normalized velocity
        return self.velocity.normalize()
    
    def calculate_attraction(self, humans):
        if not humans:
            return Vector2(0, 0)
        closest_human = min(humans, key=lambda human: self.position.distance_to(human.position))
        attraction = (closest_human.position - self.position) * ZOMBIE_ATTRACTION_WEIGHT
        return attraction


class AgentSprite(Sprite):
    def __init__(self, agent, surface, color):
        super().__init__()
        self.image = surface
        self.agent = agent
        self.color = color
        self.highlight_color = HIGHLIGHT_COLOR
        self.radius = AGENT_RADIUS
        # New attribute for highlight size
        self.highlight_radius = AGENT_RADIUS + 5  # Highlight is larger than the agent

    @property
    def rect(self):
        return self.image.get_rect(center=(int(self.agent.position.x), int(self.agent.position.y)))


class Simulation:

    # 1. Initialization and setup methods

    def __init__(self, num_humans=80, num_zombies=20):
        pygame.init()
        self.setup_screen()
        self.human_surface = self.create_agent_surface(HUMAN_COLOR)
        self.zombie_surface = self.create_agent_surface(ZOMBIE_COLOR)
        self.trail_map = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA).convert_alpha()
        self.trail_map.fill(BACKGROUND_COLOR)
        self.highlight_map = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA).convert_alpha()
        self.highlight_map.fill(BACKGROUND_COLOR)
        self.create_agents(num_humans, num_zombies)
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False

    def setup_screen(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

    def create_agent_surface(self, color):
        surface = pygame.Surface((AGENT_RADIUS*2, AGENT_RADIUS*2), pygame.SRCALPHA).convert_alpha()
        pygame.draw.circle(surface, color, (AGENT_RADIUS, AGENT_RADIUS), AGENT_RADIUS)
        return surface

    def create_agents(self, num_humans, num_zombies):
        self.agents = []
        self.humans = [Human(*self.random_position()) for _ in range(num_humans)]
        self.zombies = [Zombie(*self.random_position()) for _ in range(num_zombies)]
        self.agents.extend(self.humans)
        self.agents.extend(self.zombies)
        self.human_sprites = Group([AgentSprite(human, self.human_surface, HUMAN_COLOR) for human in self.humans])
        self.zombie_sprites = Group([AgentSprite(zombie, self.zombie_surface, ZOMBIE_COLOR) for zombie in self.zombies])
        self.agent_sprites = Group(self.human_sprites, self.zombie_sprites)
        
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
        # Prepare positions for separate nearest neighbor searches
        human_positions = np.array([human.position for human in self.humans])
        zombie_positions = np.array([zombie.position for zombie in self.zombies])
        
        human_nbrs = NearestNeighbors(n_neighbors=10).fit(human_positions) if len(human_positions) > 0 else None
        zombie_nbrs = NearestNeighbors(n_neighbors=10).fit(zombie_positions) if len(zombie_positions) > 0 else None

        # Pass the list to the update functions
        self.update_human_agents(human_nbrs, zombie_nbrs)
        self.update_zombie_agents(human_nbrs)

        # Convert humans to zombies after all updates
        collisions = self.check_collisions()
        human_to_convert = [human_sprite.agent for human_sprite in collisions]
        self.convert_humans_to_zombies(human_to_convert)

    def update_human_agents(self, human_nbrs, zombie_nbrs):
        for human in list(self.humans):  # Using list to avoid RuntimeError due to list modification
            if human_nbrs is not None:
                human_indices = human_nbrs.radius_neighbors(np.array([human.position]).reshape(1, -1), radius=HUMAN_DETECTION_RADIUS, return_distance=False)[0]
                human_neighbours = [self.humans[i] for i in human_indices]
            else:
                human_neighbours = []
            if zombie_nbrs is not None:
                zombie_indices = zombie_nbrs.radius_neighbors(np.array([human.position]).reshape(1, -1), radius=HUMAN_DETECTION_RADIUS, return_distance=False)[0]
                zombie_neighbours = [self.zombies[i] for i in zombie_indices]
            else:
                zombie_neighbours = []
            forces = human.calculate_forces(human_neighbours + zombie_neighbours)
            human.acceleration += forces
            human.move()
            
            human.is_highlighted = human.is_force_large(forces)

    def check_collisions(self):
        # Check for collisions between zombies and humans
        collisions = pygame.sprite.groupcollide(self.human_sprites, self.zombie_sprites, False, False,
                                                pygame.sprite.collide_circle)
        return collisions

    def convert_humans_to_zombies(self, humans_to_convert):
        for human in list(humans_to_convert):
            # Remove human from all lists and groups
            self.humans.remove(human)
            self.agents.remove(human)
            self.human_sprites.remove([s for s in self.human_sprites if s.agent == human])
            self.agent_sprites.remove([s for s in self.agent_sprites if s.agent == human])
            # Add zombie to all lists and groups
            new_zombie = Zombie(human.position.x, human.position.y)
            new_zombie_sprite = AgentSprite(new_zombie, self.zombie_surface, ZOMBIE_COLOR)
            self.zombies.append(new_zombie)
            self.agents.append(new_zombie)
            self.zombie_sprites.add(new_zombie_sprite)
            self.agent_sprites.add(new_zombie_sprite)

    def update_zombie_agents(self, human_nbrs):
        for zombie in list(self.zombies):
            if human_nbrs is not None:
                human_indices = human_nbrs.radius_neighbors(np.array([zombie.position]).reshape(1, -1), radius=ZOMBIE_DETECTION_RADIUS, return_distance=False)[0]
                human_neighbours = [self.humans[i] for i in human_indices]
            else:
                human_neighbours = []
            forces = zombie.calculate_forces(human_neighbours)
            zombie.acceleration += forces
            zombie.move()
            
            zombie.is_highlighted = zombie.is_force_large(forces)

    # 4. Render and display methods

    def render_agents(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.update_trails()
        self.update_highlights()
        self.screen.blit(self.highlight_map, (0, 0))
        self.screen.blit(self.trail_map, (0, 0))
        self.agent_sprites.update()
        self.agent_sprites.draw(self.screen)
        self.display_on_screen_info()
        pygame.display.flip()
        self.clock.tick(60)

    def update_trails(self):
        self.trail_map.fill((0, 0, 0, 10), special_flags=pygame.BLEND_RGBA_SUB)
        for agent in self.agent_sprites:
            pygame.draw.circle(self.trail_map, agent.color + (90,), (int(agent.agent.position.x), int(agent.agent.position.y)), AGENT_RADIUS)

    def update_highlights(self):
        self.highlight_map.fill(BACKGROUND_COLOR)
        for agent_sprite in self.agent_sprites:
            if agent_sprite.agent.is_highlighted:
                pygame.draw.circle(self.highlight_map, HIGHLIGHT_COLOR + (128,), agent_sprite.agent.position, AGENT_RADIUS + 5)

    def display_on_screen_info(self):
        font = pygame.font.SysFont('Arial', 20)
        self.screen.blit(font.render(f'Humans: {len(self.humans)}', False, WORD_COLOR), (10, 10))
        self.screen.blit(font.render(f'Zombies: {len(self.zombies)}', False, WORD_COLOR), (10, 30))


if __name__ == '__main__':
    Simulation().run()


"""
To improve the efficiency of your simulation, consider the following optimization strategies:

### Use Vectorized Operations for Distance Calculations
- **NumPy for Vectorized Calculations**: Utilize NumPy's vectorized operations to compute distances and other vector-based calculations instead of Python loops. This can significantly speed up operations like separation, alignment, and cohesion for agents.

### Optimize Drawing and Rendering
- **Minimize Draw Calls**: Reduce the number of draw calls by batch processing drawing operations where possible. For instance, drawing all agents of the same type in a single operation can be more efficient than individual draw calls for each agent.

### Directly Manage NumPy Arrays for Positions
- **Maintain Separate Position Arrays**: Keep separate NumPy arrays for human and zombie positions throughout the simulation. Update these arrays directly when agents move or are converted, rather than converting back and forth between lists and arrays.
- **Update Positions in Batch**: When possible, update agent positions in the arrays using batch operations. This approach is more efficient than updating positions individually but may require careful tracking of indices and conditions.

### Implementation Considerations
- **Tracking Indices**: To update positions directly in NumPy arrays, you may need to track the indices of humans and zombies within these arrays. This adds complexity but can lead to performance improvements by avoiding the reconstruction of position arrays at each simulation step.
- **Vectorized Distance Calculations**: For operations that involve computing distances between agents (e.g., nearest neighbor searches), leverage NumPy's ability to perform these calculations in a vectorized manner for efficiency gains.

Please implement some of them and leave the other to the next update.
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

"""
Combining a boid algorithm with pathfinding in a zombie apocalypse simulation is an intriguing idea! The boid algorithm can be used to model the flocking behavior of a group of entities (like zombies or survivors), while pathfinding can help these entities navigate complex environments to reach specific targets.

Here's a basic outline to combine both:

1. *Environment Setup*:
   - Design a map with obstacles, safe zones, resources, etc.
   - Place the survivors and zombies on the map.

2. *Boid Algorithm for Flocking Behavior*:
   - *Separation*: Ensure that zombies (or survivors) don't overlap with each other.
   - *Alignment*: Zombies align their direction with the average heading of their local flockmates.
   - *Cohesion*: Zombies move toward the average position of their local flockmates.
   - For zombies, you might introduce an additional behavior, like "attraction" towards noise or movement.

3. *Pathfinding*:
   - Use algorithms like A* or Dijkstra to find the shortest path to a target. This can be used when a zombie detects a survivor and tries to approach them, or when a survivor tries to reach a safe zone.
   - The pathfinding algorithm will guide the zombie or survivor around obstacles.

4. *Combine the Two*:
   - *Perception Radius*: Zombies have a radius within which they detect survivors. If a survivor is in this radius, the zombie switches from its flocking behavior to pathfinding mode to chase the survivor.
   - *Distracting Agents*: Maybe survivors can throw objects or make noises. Any zombie within a certain radius gets attracted to this point and uses pathfinding to navigate to it.
   - *Safety in Numbers*: Survivors can use flocking behaviors among themselves to stick together and evade zombies. If a survivor is isolated, they might use pathfinding to rejoin their group.

5. *Additional Considerations*:
   - *Energy/Health*: You can introduce an energy or health system. Zombies might degrade over time or when they encounter obstacles. Survivors might lose health when attacked by zombies.
   - *Environment Interaction*: As zombies flock and move, they can push objects, create noise, or even break down barriers, affecting the pathfinding algorithm.
   - *Dynamic Pathfinding*: If barriers are broken or environments change (e.g., fire spreading), the pathfinding algorithm needs to account for these changes.

6. *Optimization*:
   - Since pathfinding can be computationally expensive, especially for large numbers of agents, you might want to limit how often it's recalculated.
   - Only calculate paths for zombies in close proximity to survivors or dynamic events.

By combining boid algorithms with pathfinding, you can create a more realistic and exciting simulation where zombies exhibit both swarm behaviors and directed pursuit, and survivors have the capability to evade, group, and navigate complex terrains.
"""

# https://github.com/warownia1/PythonCollider
# https://stackoverflow.com/questions/28997397/pygame-use-of-pygame-sprite-layeredupdates
