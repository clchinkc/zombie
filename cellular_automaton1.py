class Population:
    def __init__(self, grid_size, birth_rate, death_rate, migration_rate, min_neighbors):
        self.grid_size = grid_size
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.migration_rate = migration_rate
        self.min_neighbors = min_neighbors
        self.grid = [[0 for i in range(grid_size)] for j in range(grid_size)]

    def set_cell(self, x, y, value):
        self.grid[x][y] = value

    def get_cell(self, x, y):
        return self.grid[x][y]

    def get_neighbor_count(self, x, y):
        count = 0
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if i >= 0 and i < self.grid_size and j >= 0 and j < self.grid_size and (i != x or j != y):
                    count += self.get_cell(i, j)
        return count

    def update(self):
        new_grid = [[0 for i in range(self.grid_size)] for j in range(self.grid_size)]
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


grid_size = 10
birth_rate = 0.1
death_rate = 0.05
migration_rate = 0.01
min_neighbors = 0.5

population = Population(grid_size, birth_rate, death_rate, migration_rate, min_neighbors)

# set the initial state of the cells
population.set_cell(1, 1, 1)
population.set_cell(1, 2, 1)
population.set_cell(1, 3, 1)

# simulate the dynamics of the population over multiple time steps
for i in range(100):
    population.update()

# print the updated state of the cells
for i in range(grid_size):
    for j in range(grid_size):
        print(population.get_cell(i, j), end="\t")
    print()
