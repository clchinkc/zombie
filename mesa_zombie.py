
import random

import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation


# define the model
class ZombieModel(Model):
    
    def __init__(self, zombie_speed, human_speed, width, height):
        self.zombie_speed = zombie_speed
        self.human_speed = human_speed
        self.grid = ContinuousSpace(width, height, torus=False)
        self.schedule = RandomActivation(self)
        
        # create a zombie and a human agent
        zombie = Zombie(1, self.zombie_speed, self)
        human = Human(2, self.human_speed, self)
        
        # place the agents randomly in the grid
        self.grid.place_agent(zombie, (random.uniform(0, width), random.uniform(0, height)))
        self.grid.place_agent(human, (random.uniform(0, width), random.uniform(0, height)))
        
        self.schedule.add(zombie)
        self.schedule.add(human)
    
    def step(self):
        self.schedule.step()

# define the zombie agent
class Zombie(Agent):
    
    def __init__(self, unique_id, speed, model):
        super().__init__(unique_id, model)
        self.speed = speed
    
    def move(self):
        # get the position of the human agent
        human = self.model.schedule.agents[1]
        human_pos = human.pos
        
        # calculate the distance to the human
        distance = self.pos.distance(human_pos)
        
        # if the zombie is close enough to the human, move towards the human
        if distance < self.speed:
            self.model.grid.move_agent(self, human_pos)
    
    def step(self):
        self.move()

# define the human agent
class Human(Agent):
    
    def __init__(self, unique_id, speed, model):
        super().__init__(unique_id, model)
        self.speed = speed
    
    def move(self):
        # move randomly
        dx = random.uniform(-self.speed, self.speed)
        dy = random.uniform(-self.speed, self.speed)
        new_pos = (self.pos[0]+dx, self.pos[1]+dy)
        
        # check if the new position is within the grid, and move the agent
        if self.model.grid.out_of_bounds(new_pos):
            new_pos = self.model.grid.torus_adj(self.pos)
        self.model.grid.move_agent(self, new_pos)
    
    def step(self):
        self.move()

# create the model
zombie_speed = 5
human_speed = 2
width = 10
height = 10
model = ZombieModel(zombie_speed, human_speed, width, height)

# run the model
for i in range(100):
    model.step()
    
# plot the results
zombie = model.schedule.agents[0]
human = model.schedule.agents[1]
plt.scatter(zombie.pos[0], zombie.pos[1], color='green')
plt.scatter(human.pos[0], human.pos[1], color='red')
plt.xlim(0, width)
plt.ylim(0, height)
plt.show()

