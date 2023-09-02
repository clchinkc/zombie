import math
import random
import sys

import numpy as np
import pygame
from pygame.math import Vector2


# Flocking behavior for an agent
class FlockingAgent:
    def __init__(self, position, max_speed=2, max_force=0.03):
        self.position = position
        self.velocity = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        self.acceleration = Vector2()
        self.max_speed = max_speed
        self.max_force = max_force

    def seek(self, target):
        desired = target - self.position
        desired.normalize()
        desired = desired * self.max_speed
        steer = desired - self.velocity
        steer = np.clip(steer, -self.max_force, self.max_force)
        return steer

    def apply_force(self, force):
        self.acceleration = self.acceleration + force

    def flock(self, agents):
        sep = self.separate(agents)
        ali = self.align(agents)
        coh = self.cohesion(agents)
        
        sep = sep * 1.5
        ali = ali * 1.0
        coh = coh * 1.0
        
        self.apply_force(sep)
        self.apply_force(ali)
        self.apply_force(coh)

    def separate(self, agents):
        desired_separation = 25.0
        steer = Vector2()
        count = 0
        for agent in agents:
            d = math.sqrt((self.position.x - agent.position.x)**2 + (self.position.y - agent.position.y)**2)
            if 0 < d < desired_separation:
                diff = self.position - agent.position
                diff.normalize()
                diff = diff / d
                steer = steer + diff
                count += 1
        if count > 0:
            steer = steer / count
        if steer.magnitude() > 0:
            steer.normalize()
            steer = steer * self.max_speed
            steer = steer - self.velocity
            steer = np.clip(steer, -self.max_force, self.max_force)
        return steer

    def align(self, agents):
        neighbordist = 50
        sum_vector = Vector2()
        count = 0
        for agent in agents:
            d = math.sqrt((self.position.x - agent.position.x)**2 + (self.position.y - agent.position.y)**2)
            if 0 < d < neighbordist:
                sum_vector = sum_vector + agent.velocity
                count += 1
        if count > 0:
            sum_vector = sum_vector / count
            sum_vector.normalize()
            sum_vector = sum_vector * self.max_speed
            steer = sum_vector - self.velocity
            steer = np.clip(steer, -self.max_force, self.max_force)
            return steer
        else:
            return Vector2(0, 0)

    def cohesion(self, agents):
        neighbordist = 50
        sum_vector = Vector2()
        count = 0
        for agent in agents:
            d = math.sqrt((self.position.x - agent.position.x)**2 + (self.position.y - agent.position.y)**2)
            if 0 < d < neighbordist:
                sum_vector = sum_vector + agent.position
                count += 1
        if count > 0:
            sum_vector = sum_vector / count
            return self.seek(sum_vector)
        else:
            return Vector2(0, 0)

    def update(self):
        self.velocity = self.velocity + self.acceleration
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        self.position = self.position + self.velocity
        self.acceleration = self.acceleration * 0

# Modified Agent class with flocking behavior
class Agent(FlockingAgent):
    def __init__(self, name, health, attack_power, position):
        super().__init__(position)
        self.name = name
        self.health = health
        self.attack_power = attack_power
        self.is_alive = True

# Modified Survivor class with flocking behavior
class Survivor(Agent):
    MAX_HEALTH = 100

# Modified WarriorSurvivor class with flocking behavior
class WarriorSurvivor(Survivor):
    def __init__(self, name, position):
        super().__init__(name, health=100, attack_power=10, position=position)
        self.combat_bonus = 2

class ScavengerSurvivor(Survivor):
    def __init__(self, name, position):
        super().__init__(name, health=100, attack_power=10, position=position)
        self.scavenging_bonus = 2

# Redefining the Zombie class with a basic implementation
class Zombie(Agent):
    def __init__(self, name, health, attack_power, position):
        super().__init__(name, health, attack_power, position)


# Implementing the Zombie class with flocking behavior
class FlockingZombie(Zombie, FlockingAgent):
    def __init__(self, name, health, attack_power, position):
        Zombie.__init__(self, name, health, attack_power, position)
        FlockingAgent.__init__(self, position)

    def update(self, survivors):
        self.flock(survivors)
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        self.position = self.position + self.velocity
        self.acceleration = self.acceleration * 0
        
        for survivor in survivors:
            d = math.sqrt((self.position.x - survivor.position.x)**2 + (self.position.y - survivor.position.y)**2)
            if d < 10:
                survivor.health -= self.attack_power
                if survivor.health <= 0:
                    survivor.is_alive = False
                    
        return survivors
    
# Implement the simulation
class Simulation:
    def __init__(self, width=800, height=600, num_warriors=10, num_scavengers=10, num_zombies=10):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Zombie Simulation")
        self.clock = pygame.time.Clock()
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))
        self.agents = []
        self.num_warriors = num_warriors
        self.num_scavengers = num_scavengers
        self.num_zombies = num_zombies
        self.font = pygame.font.SysFont("monospace", 16)
        self.font_bold = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 32)
        self.font_huge = pygame.font.SysFont("monospace", 48)
        self.font_color = (0, 0, 0)
        self.game_over = False
        self.wave = 1
        self.score = 0
        self.high_score = 0
        self.initialize_simulation()

    def initialize_simulation(self):
        self.agents = []
        self.game_over = False
        self.wave = 1
        self.score = 0
        self.spawn_survivors()
        self.spawn_zombies()

    def spawn_survivors(self):
        for i in range(self.num_warriors):
            self.agents.append(WarriorSurvivor("Warrior " + str(i), Vector2(random.randint(0, self.width), random.randint(0, self.height))))
        for i in range(self.num_scavengers):
            self.agents.append(ScavengerSurvivor("Scavenger " + str(i), Vector2(random.randint(0, self.width), random.randint(0, self.height))))

    def spawn_zombies(self):
        for i in range(self.num_zombies):
            self.agents.append(FlockingZombie("Zombie " + str(i), 100, 10, Vector2(random.randint(0, self.width), random.randint(0, self.height))))

    def run(self):
        while True:
            self.clock.tick(60)
            self.handle_events()
            self.update_simulation()
            self.draw_simulation()
            
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
    def update_simulation(self):
        if not self.game_over:
            self.update_agents()
            self.check_for_collisions()
            self.check_for_game_over()
        else:
            self.check_for_restart()
            
    def update_agents(self):
        survivors = []
        zombies = []
        for agent in self.agents:
            if isinstance(agent, Survivor):
                survivors.append(agent)
            elif isinstance(agent, Zombie):
                zombies.append(agent)
        for agent in self.agents:
            if isinstance(agent, Survivor):
                agent.flock(survivors)
                agent.update()
            elif isinstance(agent, Zombie):
                survivors = agent.update(survivors)
        self.agents = survivors + zombies
        self.check_for_new_wave()
        
    def check_for_new_wave(self):
        zombies = []
        for agent in self.agents:
            if isinstance(agent, Zombie):
                zombies.append(agent)
        if len(zombies) == 0:
            self.wave += 1
            self.num_zombies += 10
            self.spawn_zombies()
                
    def check_for_collisions(self):
        survivors = []
        zombies = []
        for agent in self.agents:
            if isinstance(agent, Survivor):
                survivors.append(agent)
            elif isinstance(agent, Zombie):
                zombies.append(agent)
        for survivor in survivors:
            for zombie in zombies:
                d = math.sqrt((survivor.position.x - zombie.position.x)**2 + (survivor.position.y - zombie.position.y)**2)
                if d < 10:
                    survivor.health -= zombie.attack_power
                    if survivor.health <= 0:
                        survivor.is_alive = False
                        
    def check_for_game_over(self):
        zombies = []
        for agent in self.agents:
            if isinstance(agent, Zombie):
                zombies.append(agent)
        if len(zombies) == 0:
            self.game_over = True
            
    def check_for_restart(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            self.initialize_simulation()
            
    def draw_simulation(self):
        self.screen.blit(self.background, (0, 0))
        self.draw_agents()
        self.draw_game_info()
        pygame.display.flip()
        
    def draw_agents(self):
        for agent in self.agents:
            if isinstance(agent, Survivor):
                if isinstance(agent, WarriorSurvivor):
                    color = (0, 0, 255)
                elif isinstance(agent, ScavengerSurvivor):
                    color = (0, 255, 0)
                pygame.draw.circle(self.screen, color, (int(agent.position.x), int(agent.position.y)), 10)
            elif isinstance(agent, Zombie):
                pygame.draw.circle(self.screen, (255, 0, 0), (int(agent.position.x), int(agent.position.y)), 10)
                
    def draw_game_info(self):
        if not self.game_over:
            self.draw_game_stats()
        else:
            self.draw_game_over()
            
    def draw_game_stats(self):
        self.draw_text("Wave: " + str(self.wave), self.font, self.font_color, 10, 10)
        self.draw_text("Score: " + str(self.score), self.font, self.font_color, 10, 30)
        self.draw_text("High Score: " + str(self.high_score), self.font, self.font_color, 10, 50)
        
    def draw_game_over(self):
        self.draw_text("GAME OVER", self.font_huge, self.font_color, self.width/2, self.height/2 - 48)
        self.draw_text("Score: " + str(self.score), self.font_big, self.font_color, self.width/2, self.height/2)
        self.draw_text("Press R to restart", self.font, self.font_color, self.width/2, self.height/2 + 48)
        
    def draw_text(self, text, font, color, x, y):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.center = (x, y)
        self.screen.blit(text_surface, text_rect)
        
simulation = Simulation()
simulation.run()


