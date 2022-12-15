import random
import numpy as np
import matplotlib.pyplot as plt

HUMAN = 1
ZOMBIE = 2

# Define the possible states of the grid
class Population:
    def __init__(self, grid_size, birth_rate, death_rate, migration_rate, infection_rate, min_neighbors):
        self.grid_size = grid_size
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.migration_rate = migration_rate
        self.infection_rate = infection_rate
        self.min_neighbors = min_neighbors
        self.grid = [[0 for i in range(grid_size)] for j in range(grid_size)]

        """
        self.grid = np.zeros((self.grid_size,self.grid_size))
        """
        
    def set_cell(self, x, y, value):
        self.grid[x][y] = value

    def get_cell(self, x, y):
        return self.grid[x][y]

    def get_neighbor_count(self, x, y, kind=(HUMAN or ZOMBIE)):
        count = 0
        for dx in range(x-1, x+2):
            for dy in range(y-1, y+2):
                if self.grid[x+dx][y+dy] == kind:
                    if x+dx >= 0 and x+dx < self.grid_size and y+dy >= 0 and y+dy < self.grid_size and (dx != 0 or dy != 0):
                        count += 1
        return count

    # if the surrounding cells is 2, then current cell move away from the 2
    def get_migration_pos(self, x, y):
        if self.get_neighbor_count(x, y, kind=ZOMBIE):
            for dx in range(x-1, x+2):
                for dy in range(y-1, y+2):
                    if x+dx >= 0 and x+dx < self.grid_size and y+dy >= 0 and y+dy < self.grid_size and (dx != 0 or dy != 0):
                        surrounding_cells = self.grid[x+dx][y+dy]
                        if surrounding_cells == 2:
                            return (x-dx, y-dy)
        while True:
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            if x+dx >= 0 and x+dx < self.grid_size and y+dy >= 0 and y+dy < self.grid_size and (dx != 0 or dy != 0):
                return (x+dx, y+dy)        

    # Implement the population method
    def move_individual(self):
        new_grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                neighbors = self.get_neighbor_count(x, y)
                if self.grid[x][y] == 0 and neighbors >= self.min_neighbors:
                    if random.random() < self.birth_rate:
                        new_grid[x][y] = 1
                elif self.grid[x][y] == 1:
                    if random.random() < self.death_rate:
                        new_grid[x][y] = 0
                    elif random.random() < self.migration_rate:
                        new_pos = self.get_migration_pos(x, y)
                        new_grid[new_pos[0]][new_pos[1]] = 1
                else:
                    new_grid[x][y] = self.grid[x][y]
        self.grid = new_grid
    
        """
    def move_individual(self):
        new_grid = [[0 for _ in range(self.grid_size)] for j in range(self.grid_size)]
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                neighbor_count = self.get_neighbor_count(i, j)
                cell_value = self.get_cell(i, j)

                # calculate number of births and deaths
                births = self.birth_rate * cell_value * (1 - cell_value)
                deaths = self.death_rate * cell_value

                # calculate number of individuals that migrate
                migrants = 0
                # only consider cells with at least min_neighbors for migration
                if neighbor_count >= self.min_neighbors:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if i+dx >= 0 and i+dx < self.grid_size and j+dy >= 0 and j+dy < self.grid_size and (dx != 0 or dy != 0):
                                migrants += self.migration_rate * cell_value * self.get_cell(i+dx, j+dy)

                # update cell value
                new_value = cell_value + births - deaths + migrants
                new_grid[i][j] = new_value

        self.grid = new_grid
    """
    
    # Implement the cellular automaton method        
    def spread_infection(self):
        new_grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for x in range(grid_size):
            for y in range(grid_size):
                for dx in range(x-1, x+2):
                    for dy in range(y-1, y+2):
                        if x+dx >= 0 and x+dx < self.grid_size and y+dy >= 0 and y+dy < self.grid_size and (dx != 0 or dy != 0):
                            if self.grid[x][y] == 1:
                                if random.random() < self.infection_rate * self.get_neighbor_count(x, y, kind=ZOMBIE):
                                    new_grid[x][y] = 2
        self.grid = new_grid
          
    def update(self):
        self.move_individual()
        self.spread_infection()

# print the updated state of the cells
def print_population(population):
    grid_size = population.grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            print(population.get_cell(i, j), end=" ")
        print()
        
grid_size = 10
birth_rate = 0.1
death_rate = 0.05
migration_rate = 0.1
infection_rate = 0.1
min_neighbors = 0

population = Population(grid_size, birth_rate, death_rate, migration_rate, infection_rate, min_neighbors)

# set the initial state of the cells
for i in range(grid_size):
  for j in range(grid_size):
    if np.random.random() < 0.1:
      # Place a human individual with probability 0.1
      population.set_cell(i, j, 1)

print_population(population)
print()

# simulate the dynamics of the population over multiple time steps
for i in range(100):
    population.update()

        
print_population(population)

# Update the display of the simulation
plt.imshow(population.grid, cmap='gray')
plt.show()
plt.pause(0.01)

"""
May alter the rates using other functions
May add speed of individual
"""