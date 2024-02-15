import random
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import pygame
from pygame.math import Vector2
from pygame.sprite import Group, LayeredUpdates, Sprite
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

    def calculate_friction_and_noise(self, force):
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
        super().__init__(x, y, ZOMBIE_COLOR)

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
    def __init__(self, agent, surface):
        super().__init__()
        self.image = surface
        self.agent = agent
        self.rect = self.image.get_rect(center=(int(agent.position.x), int(agent.position.y)))

    def update(self):
        self.agent.move()
        self.rect.center = (int(self.agent.position.x), int(self.agent.position.y))


class Simulation:

    # 1. Initialization and setup methods

    def __init__(self, num_humans=80, num_zombies=20):
        pygame.init()
        self.setup_screen()
        self.human_surface = self.create_agent_surface(HUMAN_COLOR)
        self.zombie_surface = self.create_agent_surface(ZOMBIE_COLOR)
        self.trail_map = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA).convert_alpha()
        self.trail_map.fill(BACKGROUND_COLOR)
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
        self.agent_sprites = LayeredUpdates([AgentSprite(agent, self.human_surface if isinstance(agent, Human) else self.zombie_surface) for agent in self.agents])
        
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

        # Initialize a list to store humans that need to be converted to zombies
        humans_to_convert = []

        # Pass the list to the update functions
        self.update_human_agents(human_nbrs, zombie_nbrs, humans_to_convert)
        self.update_zombie_agents(human_nbrs)

        # Convert humans to zombies after all updates
        self.convert_humans_to_zombies(humans_to_convert)

    def update_human_agents(self, human_nbrs, zombie_nbrs, humans_to_convert):
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

            # Check for infection and add to the list for conversion
            infected = any([human.position.distance_to(zombie.position) < ZOMBIE_INFECTION_RADIUS for zombie in zombie_neighbours])
            if infected:
                humans_to_convert.append(human)

    def convert_humans_to_zombies(self, humans_to_convert):
        for human in list(humans_to_convert):
            self.humans.remove(human)
            self.agents.remove(human)
            new_zombie = Zombie(human.position.x, human.position.y)
            new_zombie.velocity = human.velocity
            self.zombies.append(new_zombie)
            self.agents.append(new_zombie)
            self.agent_sprites.remove([s for s in self.agent_sprites if s.agent == human])
            self.agent_sprites.add(AgentSprite(new_zombie, self.zombie_surface))

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

    # 4. Render and display methods

    def render_agents(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.update_trails()
        self.screen.blit(self.trail_map, (0, 0))
        self.agent_sprites.update()
        self.agent_sprites.draw(self.screen)
        self.display_on_screen_info()
        pygame.display.flip()
        self.clock.tick(60)

    def update_trails(self):
        self.trail_map.fill((0, 0, 0, 10), special_flags=pygame.BLEND_RGBA_SUB)
        for agent in self.agents:
            pygame.draw.circle(self.trail_map, agent.color + (90,), (int(agent.position.x), int(agent.position.y)), AGENT_RADIUS)

    def display_on_screen_info(self):
        font = pygame.font.SysFont('Arial', 20)
        self.screen.blit(font.render(f'Humans: {len(self.humans)}', False, WORD_COLOR), (10, 10))
        self.screen.blit(font.render(f'Zombies: {len(self.zombies)}', False, WORD_COLOR), (10, 30))


if __name__ == '__main__':
    Simulation().run()