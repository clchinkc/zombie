import random

class Population:
  def __init__(self, grid_size, birth_rate, death_rate, migration_rate, min_neighbors):
    self.grid_size = grid_size
    self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    self.birth_rate = birth_rate
    self.death_rate = death_rate
    self.migration_rate = migration_rate
    self.min_neighbors = min_neighbors

  def set_cell(self, x, y, value):
    self.grid[x][y] = value

  def get_cell(self, x, y):
    return self.grid[x][y]

  def get_neighbor_count(self, x, y):
    neighbor_count = 0
    for dx in [-1, 0, 1]:
      for dy in [-1, 0, 1]:
        if dx == 0 and dy == 0:
          continue
        x2 = x + dx
        y2 = y + dy
        if x2 >= 0 and x2 < self.grid_size and y2 >= 0 and y2 < self.grid_size:
          neighbor_count += self.grid[x2][y2]
    return neighbor_count

  def update(self):
    new_grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
    for x in range(self.grid_size):
      for y in range(self.grid_size):
        neighbors = self.get_neighbor_count(x, y)
        if self.grid[x][y] == 0 and neighbors >= self.min_neighbors:
          if random.random() < self.birth_rate(neighbors):
            new_grid[x][y] = 1
        elif self.grid[x][y] == 1:
          if random.random() < self.death_rate(neighbors):
            new_grid[x][y] = 0
          elif random.random() < self.migration_rate(neighbors):
            new_pos = self.get_migration_pos(x, y)
            new_grid[new_pos[0]][new_pos[1]] = 1
        else:
          new_grid[x][y] = self.grid[x][y]
    self.grid = new_grid

  def get_migration_pos(self, x, y):
    while True:
      dx = random.choice([-1, 0, 1])
      dy = random.choice([-1, 0, 1])
      if dx == 0 and dy == 0:
        continue
      x2 = x + dx
      y2 = y + dy
      if x2 >= 0 and x2 < self.grid_size and y2 >= 0 and y2 < self.grid_size:
        return (x2, y2)
    
# Create a population with a grid size of 10 cells and a birth rate of 0.1
pop = Population(10, 0.1, 0.1, 0.1, 0)

# Set the initial state of the grid
pop.set_cell(4, 4, 1)
pop.set_cell(4, 5, 1)
pop.set_cell(5, 4, 1)

# Run the simulation for 100 time steps
for _ in range(100):
  pop.update()

# Print the state of the grid after the simulation
for x in range(pop.grid_size):
  for y in range(pop.grid_size):
    print(pop.get_cell(x, y), end=" ")
  print()