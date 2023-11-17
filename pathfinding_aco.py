import numpy as np
from matplotlib import pyplot as plt


class Graph:
    def __init__(self, grid_size, obstacles):
        self.grid_size = grid_size
        self.obstacles = set(obstacles)
        self.pheromones = {}

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Including diagonals
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and (nx, ny) not in self.obstacles:
                safe = True
                # Check for proximity to obstacles
                for ox, oy in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                    if (nx + ox, ny + oy) in self.obstacles:
                        safe = False
                        break
                if safe:
                    neighbors.append((nx, ny))
        return neighbors

    def pheromone_level(self, node1, node2):
        return self.pheromones.get((node1, node2), 1.0)  # Default pheromone level

    def add_pheromone(self, node1, node2, amount):
        self.pheromones[(node1, node2)] = self.pheromones.get((node1, node2), 1.0) + amount

    def evaporate_pheromones(self, decay_rate):
        for edge in self.pheromones:
            self.pheromones[edge] *= (1 - decay_rate)

    def distance(self, node1, node2):
        return np.linalg.norm(np.array(node1) - np.array(node2))

class AntColony:
    def __init__(self, graph, n_ants, n_iterations, decay, local_evaporation, alpha=1, beta=1):
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.local_evaporation = local_evaporation
        self.alpha = alpha
        self.beta = beta

    def run(self, start, goal):
        shortest_path = None
        shortest_path_length = float('inf')

        for _ in range(self.n_iterations):
            paths = self.generate_paths(start, goal)
            self.update_pheromones(paths)

            for path, length in paths:
                if length < shortest_path_length:
                    shortest_path = path
                    shortest_path_length = length

        return shortest_path, shortest_path_length

    def generate_paths(self, start, goal):
        paths = []
        for _ in range(self.n_ants):
            path, length = self.generate_path(start, goal)
            paths.append((path, length))
        return paths

    def generate_path(self, start, goal):
        path = [start]
        current = start
        while current != goal:
            next_node = self.select_next_node(current, goal)
            path.append(next_node)
            self.graph.add_pheromone(current, next_node, -self.local_evaporation * self.graph.pheromone_level(current, next_node))
            current = next_node

        length = self.path_length(path) + self.obstacle_proximity_penalty(path)
        return path, length

    def obstacle_proximity_penalty(self, path):
        penalty = 0
        for node in path:
            min_dist = min(self.graph.distance(node, obs) for obs in self.graph.obstacles)
            penalty += 1 / (min_dist + 1e-5)  # Adding a small constant to avoid division by zero
        normalized_penalty = penalty / len(path)  # Normalize the penalty
        return normalized_penalty

    def select_next_node(self, current, goal):
        neighbors = self.graph.get_neighbors(current)
        pheromones = np.array([max(self.graph.pheromone_level(current, neighbor), 1e-5) for neighbor in neighbors])  # Avoid zero or negative values

        # Avoid division by zero in heuristic
        heuristic = np.array([1.0 / (self.graph.distance(neighbor, goal) + 1e-5) for neighbor in neighbors])

        probabilities = pheromones ** self.alpha * heuristic ** self.beta
        prob_sum = probabilities.sum()
        if prob_sum > 0:
            probabilities /= prob_sum
        else:
            probabilities = np.full(len(neighbors), 1.0 / len(neighbors))  # Equal probabilities if sum is zero

        return neighbors[np.random.choice(len(neighbors), p=probabilities)]

    def update_pheromones(self, paths):
        for path, length in paths:
            for i in range(len(path) - 1):
                self.graph.add_pheromone(path[i], path[i + 1], 1.0 / length)
        self.graph.evaporate_pheromones(self.decay)

    def path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.graph.distance(path[i], path[i + 1])
        return length


def visualize_path(grid_size, obstacles, path):
    # Create a grid representation
    grid = np.zeros(grid_size)
    for obstacle in obstacles:
        grid[obstacle] = -1  # Mark obstacles

    # Mark the path
    for node in path:
        grid[node] = 1

    # Mark the start and goal
    start, goal = path[0], path[-1]
    grid[start] = 2
    grid[goal] = 3

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.T, cmap='viridis', origin='lower')
    plt.colorbar(label='Status', ticks=[-1, 0, 1, 2, 3])
    plt.clim(-1.5, 3.5)
    plt.title('Ant Colony Path Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(False)
    plt.show()

# Example Usage
grid_size = (30, 30)
obstacles = [(x, y) for x in range(10, 20) for y in range(10, 20)]  # Example obstacle
graph = Graph(grid_size, obstacles)
ant_colony = AntColony(graph, n_ants=10, n_iterations=100, decay=0.1, local_evaporation=0.01, alpha=1, beta=1)

start = (0, 0)
goal = (25, 25)
path, length = ant_colony.run(start, goal)
visualize_path(grid_size, obstacles, path)