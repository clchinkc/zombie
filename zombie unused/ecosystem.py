
import random

import numpy as np
import pygame


class Agent(pygame.sprite.Sprite):
    def __init__(self, x, y, speed, health):
        super().__init__()
        self.image = pygame.Surface((10, 10))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.speed = speed
        self.dx = 0
        self.dy = 0
        self.health = health

    def update(self, food_group):
        if self.health <= 50:
            food_in_range = pygame.sprite.spritecollide(self, food_group, False, pygame.sprite.collide_circle)
            if food_in_range:
                nearest_food = min(food_in_range, key=lambda f: np.sqrt((f.rect.centerx - self.rect.centerx) ** 2 + (f.rect.centery - self.rect.centery) ** 2))
                self.move_to(nearest_food)
                self.move(self.dx, self.dy)
            else:
                self.move(self.dx, self.dy)
        else:
            self.move(self.dx, self.dy)

        # Check for collisions between agents and food, and eat food if close enough
        for food in food_group:
            if self.collide(food) and food.amount > 0:
                self.eat(food.amount)
                food.amount = 0

        # Check for collisions with the screen edges
        if self.rect.left < 0 or self.rect.right > 800:
            self.dx = -self.dx
        if self.rect.top < 0 or self.rect.bottom > 600:
            self.dy = -self.dy

        # Randomly change agent direction
        self.randomly_change_direction()

    def randomly_change_direction(self, gamma=0.1, sigma=0.1):
        # gamma is friction coefficient
        # sigma is noise strength
        # calculate new position and velocity using Langevin equation
        dvx = -gamma*self.dx + sigma*np.random.normal(0, 1)
        dvy = -gamma*self.dy + sigma*np.random.normal(0, 1)
        self.dx += dvx
        self.dy += dvy

    def move(self, dx, dy):
        self.rect.move_ip(dx * self.speed, dy * self.speed)

    def move_to(self, food):
        dx = food.rect.centerx - self.rect.centerx
        dy = food.rect.centery - self.rect.centery
        distance = np.sqrt(dx * dx + dy * dy)
        if distance > 0:
            self.dx = dx / distance
            self.dy = dy / distance

    def eat(self, food):
        self.health += food

    def reproduce(self, other_agent):
        return Agent(self.rect.centerx + random.randint(-10, 10), self.rect.centery + random.randint(-10, 10),
                    (self.speed + other_agent.speed) / 2, 50)

    def collide(self, other_sprite):
        return self.rect.colliderect(other_sprite.rect)

    def draw(self, screen):
        agent_surface = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.circle(agent_surface, (0, 255, 0), self.rect.center, 10)
        screen.blit(agent_surface, self.rect)


class Food(pygame.sprite.Sprite):
    def __init__(self, x, y, amount):
        super().__init__()
        self.image = pygame.Surface((5, 5))
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.amount = amount

    def collide(self, other_sprite):
        return self.rect.colliderect(other_sprite.rect)
    
    def draw(self, screen):
        food_surface = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.circle(food_surface, (255, 0, 0), self.rect.center, 5)
        screen.blit(food_surface, self.rect)

class Simulation:
    def __init__(self, num_agents, world_size, food_amount):
        self.agents = pygame.sprite.Group()
        self.food = pygame.sprite.Group()
        for i in range(num_agents):
            agent = Agent(random.randint(0, world_size[0]), random.randint(0, world_size[1]), random.uniform(0.5, 1.5), 50)
            self.agents.add(agent)
        for i in range(num_agents):
            food = Food(random.randint(0, world_size[0]), random.randint(0, world_size[1]), food_amount)
            self.food.add(food)
        self.world_size = world_size
        self.eat_distance = 10
        self.detect_distance = 100
        self.new_agents = []
        self.dead_agents = []
        self.hunger_threshold = 50
        self.change_direction_prob = 0.1

    def move_agents(self):
        for agent in self.agents:
            if agent.health <= self.hunger_threshold:
                food_in_range = pygame.sprite.spritecollide(agent, self.food, False, pygame.sprite.collide_circle)
                if len(food_in_range) > 0:
                    nearest_food = min(food_in_range, key=lambda f: np.sqrt((f.rect.centerx - agent.rect.centerx) ** 2 + (f.rect.centery - agent.rect.centery) ** 2))
                    agent.move_to(nearest_food)
                    agent.move(agent.dx, agent.dy)
                else:
                    agent.move(agent.dx, agent.dy)
            else:
                agent.move(agent.dx, agent.dy)

            # create Rect objects for each agent and check for collisions with the screen edges
            agent_rect = pygame.Rect(agent.rect)
            if agent_rect.left < 0:
                agent_rect.left = 0
                agent.dx = -agent.dx
            elif agent_rect.right > self.world_size[0]:
                agent_rect.right = self.world_size[0]
                agent.dx = -agent.dx

            if agent_rect.top < 0:
                agent_rect.top = 0
                agent.dy = -agent.dy
            elif agent_rect.bottom > self.world_size[1]:
                agent_rect.bottom = self.world_size[1]
                agent.dy = -agent.dy

            # check for collisions between agents and food, and eat food if close enough
            for food in self.food:
                food_rect = pygame.Rect(food.rect)
                if agent_rect.colliderect(food_rect):
                    agent.eat(food.amount)
                    food.amount = 0

            # randomly change agent direction
            if random.random() < self.change_direction_prob:
                agent.dx = random.uniform(-1, 1)
                agent.dy = random.uniform(-1, 1)

            # check if agent has died from hunger
            if agent.health <= 0:
                self.dead_agents.append(agent)

        # remove dead agents from agent list
        for agent in self.dead_agents:
            self.agents.remove(agent)
        self.dead_agents.clear()


    def reproduce_agents(self):
        for agent1 in self.agents:
            for agent2 in self.agents:
                if agent1 != agent2 and pygame.sprite.collide_circle(agent1, agent2) and agent1.health > 50 and agent2.health > 50 and random.random() < 0.1:
                    new_agent = agent1.reproduce(agent2)
                    self.new_agents.append(new_agent)
                    agent1.health -= 50
                    agent2.health -= 50

    def update(self):
        self.move_agents()
        self.reproduce_agents()
        self.agents.add(self.new_agents)
        self.new_agents.clear()

        self.food = pygame.sprite.Group([f for f in self.food if f.amount > 0])
        for i in self.food:
            if random.random() < 0.0001:
                i.amount -= 1
        for i in self.agents:
            if random.random() < 0.0001:
                i.health -= 1

class Game:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((800, 600))
        self.font = pygame.font.SysFont('Arial', 24)
        self.simulation = Simulation(10, (800, 600), 10)

    def run(self):
        running = True
        paused = False
        key_actions = {
            pygame.K_SPACE: lambda: setattr(self, 'paused', not paused),
            pygame.K_ESCAPE: lambda: setattr(self, 'running', False),
            pygame.K_UP: lambda: self.simulation.agents.add(Agent(random.randint(0, 800), random.randint(0, 600), random.uniform(0.5, 1.5), 50)),
            pygame.K_DOWN: lambda: self.simulation.agents.remove(self.simulation.agents.sprites()[-1]) if len(self.simulation.agents) > 1 else None,
            pygame.K_RIGHT: lambda: self.simulation.food.add(Food(random.randint(0, 800), random.randint(0, 600), 10)),
            pygame.K_LEFT: lambda: self.simulation.food.remove(self.simulation.food.sprites()[-1]) if len(self.simulation.food) > 1 else None
        }

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key_actions.get(event.key, lambda: None)()

            if not paused:
                self.simulation.update()

            self.screen.fill((255, 255, 255))

            self.simulation.agents.draw(self.screen)
            self.simulation.food.draw(self.screen)

            text = self.font.render(f"Agents: {len(self.simulation.agents)}  Food: {len(self.simulation.food)}", True, (0, 0, 0))
            self.screen.blit(text, (10, 10))

            text = self.font.render("Press Space to Pause/Resume, Arrow Keys to Add/Remove Agents/Food", True, (0, 0, 0))
            self.screen.blit(text, (10, 570))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()

"""
Here are a few suggestions for optimizing this code:

Use sprite groups to store and manage sprites. This can improve performance by reducing the number of times that Pygame has to iterate over all of the sprites in the game.

Use dirty rectangles to only update the parts of the screen that have changed. This can improve performance by reducing the number of times that Pygame has to redraw the entire screen.

Use blitting instead of pygame.draw.rect() to draw images to the screen. This is a fast way to draw images to the screen.

Use pygame.time.Clock() to limit the frame rate of the game. This can improve performance by preventing the game from running too fast and using up too much CPU resources.

Use the pygame.sprite.Group.draw method to draw all of the sprites in a group. This is more efficient than drawing each sprite individually.

Use the pygame.display.flip method to update the display. This is more efficient than calling the pygame.display.update method for each sprite.

Use pygame.sprite.collide_circle() to check for collisions between circles. This function is more efficient than pygame.sprite.collide_rect() when checking for collisions between circles. This is because it only needs to compare the radii of the circles, rather than the entire rectangles.

Use pygame.sprite.spritecollide() to check for collisions between a sprite and a group of sprites. This function is more efficient than using pygame.sprite.collide_rect_list().

Use pygame.sprite.groupcollide() to check for collisions between two groups of sprites. This function is more efficient than using pygame.sprite.collide_rect_list_list().

Use pygame.sprite.Group.remove() to remove sprites from a group. This function is more efficient than using del to remove sprites from a group.

Use pygame.Surface.convert() to convert a Surface to a faster format, such as a 16-bit or 8-bit Surface. This can improve performance by reducing the amount of time that Pygame has to spend rendering the Surface.

Use pygame.Surface.convert_alpha() to convert a Surface to a format that supports transparency. This can improve performance by reducing the amount of time that Pygame has to spend blending the Surface with the background.

Use pygame.Surface.get_rect() to get the bounding rectangle of a Surface. This can improve performance by reducing the number of times that Pygame has to calculate the bounding rectangle of a Surface.

Use pygame.sprite.spritecollide() to efficiently check for collisions between sprites. This can improve performance by reducing the number of times that Pygame has to iterate over all of the sprites in the game.

Use pygame.sprite.spritecollide_mask() to efficiently check for collisions between sprites that have masks. This can improve performance by reducing the number of times that Pygame has to do a pixel-by-pixel collision check.

Use numpy arrays instead of lists for storing agent and food objects. Numpy arrays are more efficient for numerical operations and can be used for more efficient collision detection.

Use a Quadtree data structure to efficiently detect collisions between agents and food objects.

"""

# https://www.youtube.com/watch?v=r_It_X7v-1E
# https://github.com/SebLague/Ecosystem-2/tree/master
# https://github.com/Emergent-Behaviors-in-Biology/community-simulator/blob/master/Tutorial.ipynb

