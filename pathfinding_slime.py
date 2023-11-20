import matplotlib.pyplot as plt
import numpy as np


class SlimeMoldPathfinder:
    def __init__(self, grid_size, start, end, obstacles, n_iterations=100):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size)
        self.start = start
        self.end = end
        self.n_iterations = n_iterations
        self.pheromone_map = np.zeros(grid_size)
        self.current_position = start
        self.obstacles = obstacles

        # Initialize pheromones at the starting position and obstacles
        self.pheromone_map[start] = 1
        for obs in obstacles:
            self.grid[obs] = -1  # Marking obstacles

    def update_pheromones(self):
        # Diffuse and evaporate pheromones
        kernel = np.array([[0.1, 0.2, 0.1],
                           [0.2, 1.0, 0.2],
                           [0.1, 0.2, 0.1]])
        self.pheromone_map = self.convolve(self.pheromone_map, kernel)
        self.pheromone_map *= 0.99  # Evaporation
        # Ensure obstacles don't have pheromones
        self.pheromone_map[self.grid == -1] = 0

    def move(self):
        # Move the mold towards the area of highest pheromone concentration
        x, y = self.current_position
        neighborhood = self.pheromone_map[max(0, x-1):x+2, max(0, y-1):y+2]
        next_move = np.unravel_index(neighborhood.argmax(), neighborhood.shape)
        new_position = (max(0, x-1) + next_move[0], max(0, y-1) + next_move[1])

        # Move if the new position is not an obstacle
        if self.grid[new_position] != -1:
            self.current_position = new_position
            # Update pheromones at the new position
            self.pheromone_map[self.current_position] += 1

    def find_path(self):
        for _ in range(self.n_iterations):
            self.update_pheromones()
            self.move()

            # Terminate if the end is reached
            if self.current_position == self.end:
                print("Target reached!")
                break

        return self.pheromone_map

    @staticmethod
    def convolve(array, kernel):
        # Convolution for pheromone diffusion
        padded = np.pad(array, ((1, 1), (1, 1)), mode='constant')
        result = np.zeros_like(array)
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                result[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
        return result

# Example usage
grid_size = (20, 20)
start = (0, 0)
end = (19, 19)
obstacles = [(5, 5), (5, 6), (6, 5), (6, 6)]  # Some obstacles

pathfinder = SlimeMoldPathfinder(grid_size, start, end, obstacles)
pheromone_map = pathfinder.find_path()

# Visualize the pheromone map
plt.imshow(pheromone_map, cmap='hot')
plt.colorbar()
plt.show()