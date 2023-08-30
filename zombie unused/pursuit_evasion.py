
import time

import numpy as np
import pygame


class Agent:
    def __init__(self, position, velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.best_position = np.copy(self.position)
        self.best_score = np.inf

    def move(self):
        self.position = self.position + self.velocity

class Pursuer(Agent):
    def __init__(self, position, velocity):
        super().__init__(position, velocity)
        self.color = pygame.Color('blue')

class Evader(Agent):
    def __init__(self, position, velocity):
        super().__init__(position, velocity)
        self.color = pygame.Color('red')
        self.direction = np.random.uniform(-1, 1, 2)

    def move(self):
        self.position = self.position + self.direction
        if np.random.rand() < 0.1:
            self.direction = np.random.uniform(-1, 1, 2)

class Game:
    def __init__(self, num_agents, max_iterations, fitness_func, grid_size):
        self.grid_size = grid_size
        self.fitness_func = fitness_func
        self.agents = [Pursuer(self.random_position(), [0, 0]) for _ in range(num_agents)]
        self.agents.append(Evader(self.random_position(), [0, 0]))
        self.best_position = None
        self.best_fitness = np.inf
        self.max_iterations = max_iterations

    def random_position(self):
        if getattr(self, 'agents', None) is None:
            return np.random.uniform(0, self.grid_size, 2)
        position = np.random.uniform(0, self.grid_size, 2)
        while any(np.allclose(position, agent.position, atol=1.0) for agent in self.agents):
            position = np.random.uniform(0, self.grid_size, 2)
        return position

    def run(self):
        for i in range(self.max_iterations):
            for agent in self.agents:
                fitness = self.fitness_func(agent.position)

                if fitness < agent.best_score:
                    agent.best_score = fitness
                    agent.best_position = np.copy(agent.position)

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_position = np.copy(agent.position)

                if isinstance(agent, Pursuer):
                    phi1 = np.random.uniform(0, 1)
                    phi2 = np.random.uniform(0, 1)
                    agent.velocity = agent.velocity + phi1 * (agent.best_position - agent.position) + phi2 * (self.best_position - agent.position)
                    agent.velocity = np.clip(agent.velocity, -1, 1)

                agent.move()
                agent.position = np.clip(agent.position, 0, self.grid_size - 1)

                self.render(agent, i)

            time.sleep(1)

    def render(self, agent, i):
        window.fill(pygame.Color('black'))

        for ag in self.agents:
            color = pygame.Color('green') if ag == agent else ag.color
            pygame.draw.circle(window, color, (int(ag.position[0]*grid_cell_size), int(ag.position[1]*grid_cell_size)), 10 if ag != agent else 15)

        text = font.render('Iteration: %d' % i, True, pygame.Color('white'))
        window.blit(text, (20, 20))

        text = font.render('Min distance: %.2f' % self.best_fitness, True, pygame.Color('white'))
        window.blit(text, (20, 60))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

def fitness_function(position):
    evader_position = game.agents[-1].position
    distance = np.linalg.norm(position - evader_position)
    return distance

grid_size = 10
grid_cell_size = 50
window_size = grid_cell_size * grid_size

pygame.init()
window = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption('Pursuit Evasion Game')
font = pygame.font.Font(None, 36)

num_agents = 3
max_iterations = 10
game = Game(num_agents, max_iterations, fitness_function, grid_size)
game.run()


"""
Future improvements:
Introduce a vision range for Pursuer can only "see" Evader within a certain vision_range and move towards them.
If a Pursuer is close enough to a Evader, the Pursuer "kills" the Evader.
Introduce obstacles in the game environment that the pursuers and evader must navigate around and adjust the behaviour of both Pursuer and Evader to plan their movements. The fitness function would also need to be adjusted accordingly.
Introduce pathfinding so that zombies and humans can navigate around obstacles.
Make humans smarter, maybe group together or run away more strategically.
Introduce a safe zone for humans. If humans reach this zone, they're safe.
Maybe introduce a time factor. Humans win if they avoid zombies for a certain period. Zombies win if they infect all humans before time runs out.
"""
