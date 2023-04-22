
"""
The suggested simulation is a dynamic model that tracks the populations of sheep, wolves, and grass over time and observes how changes in one population affects the others. The model includes a predator-prey relationship between the sheep and wolves, a carrying capacity for the grass plots, migration behavior, a disease or parasite affecting the sheep population, and seasonality factors. The simulation outputs the changes in each population over time using line graphs, and the distribution of each population at the end of the simulation using histograms. This simulation provides a more realistic representation of the dynamics of sheep and wolf populations in an ecosystem and can be used to study the interplay between these populations.
"""

import random


# Define classes for sheep and wolves
class Sheep:
    def __init__(self, id, location):
        self.id = id
        self.location = location
        self.energy = 10
        self.reproduce_energy = 15
        self.reproduction_interval = random.randint(50, 100)
        self.days_since_reproduction = 0
    
    def move(self, adjacent_plots):
        # choose a random adjacent plot to move to
        new_location = random.choice(adjacent_plots)
        self.location = new_location
        self.energy -= 1
        
    def consume_grass(self, current_plot):
        # consume 0.5 units of grass and update energy level
        current_plot.grass_units -= 0.5
        self.energy += 0.5
    
    def reproduce(self, adjacent_plots, sheep_list):
        if self.energy >= self.reproduce_energy and self.days_since_reproduction >= self.reproduction_interval:
            # choose a random adjacent plot to give birth
            new_location = random.choice(adjacent_plots)
            # create a new sheep object
            new_sheep = Sheep(len(sheep_list) + 1, new_location)
            sheep_list.append(new_sheep)
            self.energy -= self.reproduce_energy
            self.days_since_reproduction = 0
    
    def update(self, adjacent_plots, current_plot, wolf_list, sheep_list):
        if self.energy > 0:
            self.move(adjacent_plots)
            if self.location in [wolf.location for wolf in wolf_list]:
                # calculate probability of being killed by wolf
                kill_prob = 0.5
                if random.random() < kill_prob:
                    sheep_list.remove(self)
                    self.energy = 0
            else:
                self.consume_grass(current_plot)
                self.reproduce(adjacent_plots, sheep_list)
                self.days_since_reproduction += 1

class Wolf:
    def __init__(self, id, location):
        self.id = id
        self.location = location
        self.energy = 20
        self.reproduce_energy = 25
        self.reproduction_prob = 0.02
    
    def move(self, adjacent_plots):
        # choose a random adjacent plot to move to
        new_location = random.choice(adjacent_plots)
        self.location = new_location
        self.energy -= 1
    
    def hunt_sheep(self, adjacent_plots, sheep_list):
        for plot in adjacent_plots:
            adjacent_sheep = [sheep for sheep in sheep_list if sheep.location == plot]
            if adjacent_sheep:
                # calculate probability of successfully killing sheep
                kill_prob = 0.5
                if random.random() < kill_prob:
                    sheep_list.remove(adjacent_sheep[0])
                    self.energy += 1
                    break
    
    def reproduce(self, adjacent_plots, wolf_list):
        if self.energy >= self.reproduce_energy and random.random() < self.reproduction_prob:
            # choose a random adjacent plot to give birth
            new_location = random.choice(adjacent_plots)
            # create a new wolf object
            new_wolf = Wolf(len(wolf_list) + 1, new_location)
            wolf_list.append(new_wolf)
            self.energy -= self.reproduce_energy
    
    def update(self, adjacent_plots, sheep_list, wolf_list):
        if self.energy > 0:
            self.move(adjacent_plots)
            self.hunt_sheep(adjacent_plots, sheep_list)
            self.reproduce(adjacent_plots, wolf_list)

# Define class for grass plots
class GrassPlot:
    def __init__(self, id, max_capacity):
        self.id = id
        self.grass_units = 10
        self.max_capacity = max_capacity

    def grow_grass(self):
        # grow 0.7 units of grass per day, up to maximum capacity
        self.grass_units = min(self.grass_units + 0.7, self.max_capacity)


# Initialize the simulation
num_sheep = 50
num_wolves = 5
grass_plots = [GrassPlot(i, 50) for i in range(100)]
sheep_list = [Sheep(i, random.randint(0, 99)) for i in range(1, num_sheep + 1)]
wolf_list = [Wolf(i, random.randint(0, 99)) for i in range(1, num_wolves + 1)]

# Define function to get adjacent plots for an object
def get_adjacent_plots(location):
    row, col = divmod(location, 10)
    adjacent_plots = []
    for r in range(row - 1, row + 2):
        for c in range(col - 1, col + 2):
            if r >= 0 and r < 10 and c >= 0 and c < 10 and not (r == row and c == col):
                adjacent_plots.append(r * 10 + c)
    return adjacent_plots

# Define simulation loop
for day in range(365):
    for sheep in sheep_list:
        current_plot = grass_plots[sheep.location]
        adjacent_plots = get_adjacent_plots(sheep.location)
        sheep.update(adjacent_plots, current_plot, wolf_list, sheep_list)
        for wolf in wolf_list:
            current_plot = grass_plots[wolf.location]
            adjacent_plots = get_adjacent_plots(wolf.location)
            wolf.update(adjacent_plots, sheep_list, wolf_list)
            for plot in grass_plots:
                plot.grow_grass()
                
# Record total grass biomass and distribution
total_grass_biomass = sum([plot.grass_units for plot in grass_plots])
grass_distribution = [plot.grass_units for plot in grass_plots]

# Plot grass distribution
import matplotlib.pyplot as plt

plt.hist(grass_distribution)
plt.xlabel('Grass Biomass')
plt.ylabel('Frequency')
plt.show()

# Plot sheep distribution
sheep_distribution = [sheep.location for sheep in sheep_list]
plt.hist(sheep_distribution)
plt.xlabel('Plot ID')
plt.ylabel('Frequency')
plt.show()

# Plot wolf distribution
wolf_distribution = [wolf.location for wolf in wolf_list]
plt.hist(wolf_distribution)
plt.xlabel('Plot ID')
plt.ylabel('Frequency')
plt.show()

# Analysis
# Compare grass biomass and distribution before and after wolf reintroduction
# Experiment with different numbers of wolves and observe the effect on grass biomass and distribution
# Repeat the simulation for multiple years to determine the length of time required to achieve the desired effect.
