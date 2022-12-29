# Description: Cellular Automaton Model of Zombie Apocalypse at School
import random
import numpy as np
import matplotlib.pyplot as plt

EMPTY = 0
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
        self.grid = np.zeros((self.grid_size,self.grid_size), dtype=int)
        neighbors = grid[max(row - 1, 0):min(row + 2, ROWS), max(col - 1, 0):min(col + 2, COLS)]
        new_state = update(state, neighbors)
        new_grid[row][col] = new_state
        """

    # Implement the population method
    def move_individual(self):
        new_grid = [[0 for _ in range(self.grid_size)]
                    for _ in range(self.grid_size)]
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                neighbors = self.get_neighbors(x, y)

                if self.grid[x][y] == 0 and neighbors >= self.min_neighbors:
                    if random.random() < self.migration_rate * sum(1 if neighbor == 1 else 0 for neighbor in neighbors):
                        # migrate from random neighbor
                        while True:
                            dx = random.randint(-1, 1)
                            dy = random.randint(-1, 1)
                            if dx != 0 and dy != 0 and self.grid[x + dx][y + dy] == 1:
                                self.grid[x + dx][y + dy] = 0
                                new_grid[x][y] = 1
                                break
                            else:
                                continue
                    else:
                        continue
                elif self.grid[x][y] == 0:
                    if random.random() < (self.birth_rate * (1 - self.birth_rate)):
                        new_grid[x][y] = 1
                    else:
                        new_grid[x][y] = 0
                elif self.grid[x][y] == 1:
                    if random.random() < self.death_rate:
                        new_grid[x][y] = 0
                    elif random.random() < self.infection_rate * sum(1 if neighbor == 2 else 0 for neighbor in neighbors):
                        new_grid[x][y] = 2
                    else:
                        new_grid[x][y] = 1
                elif self.grid[x][y] == 2:
                    if random.random() < self.death_rate:
                        new_grid[x][y] = 0
                    else:
                        new_grid[x][y] = 2
                else:
                    new_grid[x][y] = self.grid[x][y]
        self.grid = new_grid


"""
May alter the rates using other functions
May add speed of individual
May use cell class to represent some infection rate danger value etc based on neighboring cells
"""
"""
One potential improvement to the CA model could be to add more complex rules for transitioning between states. 
For example, the model could incorporate additional factors 
such as the age or health of the students and zombies, 
as well as the availability of weapons or other resources. 
This could allow for more realistic and nuanced simulations of the zombie apocalypse at school.

Another potential improvement could be to add more advanced visualization techniques, 
such as color-coding the cells to represent different states or using animation to show the progression of the simulation over time. 
This could make it easier for users to understand and interpret the results of the simulation.

Additionally, the model could be expanded to include more detailed information about the layout of the school, 
such as the locations of classrooms, doors, and other features. 
This could allow for more accurate simulations of the movement 
and interactions of students, teachers, and zombies within the school environment.
"""
