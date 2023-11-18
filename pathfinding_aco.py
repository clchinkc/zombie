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

    def add_pheromone(self, node1, node2, amount, local_evaporation_rate=0):
        self.pheromones[(node1, node2)] = (self.pheromones.get((node1, node2), 1.0) + amount) * (1 - local_evaporation_rate)

    def evaporate_pheromones(self, decay_rate):
        for edge in self.pheromones:
            self.pheromones[edge] *= (1 - decay_rate)

    def evaporate_edge_pheromone(self, node1, node2, decay_rate):
        if (node1, node2) in self.pheromones:
            self.pheromones[(node1, node2)] *= (1 - decay_rate)

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
        self.previous_success_rates = []
        self.previous_avg_path_lengths = []

    def run(self, start, goal):
        shortest_path = None
        shortest_path_length = float('inf')

        for iteration in range(self.n_iterations):
            paths = self.generate_paths(start, goal)
            self.update_pheromones(paths)
            
            current_success_rate = self.calculate_success_rate(paths)
            self.previous_success_rates.append(current_success_rate)
            current_avg_path_length = self.calculate_average_path_length(paths)
            self.previous_avg_path_lengths.append(current_avg_path_length)

            # Adaptive Parameter Tuning
            if iteration > 0:
                self.adapt_parameters(iteration, current_success_rate, current_avg_path_length)

            for path, length in paths:
                if length < shortest_path_length:
                    shortest_path = path
                    shortest_path_length = length

        return shortest_path, shortest_path_length

    def calculate_success_rate(self, paths):
        success_count = 0
        for path, length in paths:
            if length < float('inf'):
                success_count += 1
        return success_count / len(paths)

    def calculate_average_path_length(self, paths):
        total_length = sum(length for _, length in paths)
        return total_length / len(paths)

    def calculate_diversity_factor(self):
        # Implement the logic to calculate the diversity of paths
        # Calculate the standard deviation of path lengths as an example
        path_length_std = np.std(self.previous_avg_path_lengths)
        # Normalize and invert the standard deviation to get the diversity factor
        # A higher std indicates lower diversity (need for more exploitation), and vice versa
        diversity_factor = 1 / (1 + path_length_std)

        return max(0.8, min(diversity_factor, 1.2))

    def adapt_parameters(self, iteration, current_success_rate, current_avg_path_length, momentum_window=5):
        if iteration < momentum_window:
            return  # Ensure enough data points for momentum calculation

        # Calculate diversity factor for current iteration
        diversity_factor = self.calculate_diversity_factor()

        # Calculate average changes over the momentum window
        avg_success_rate_change = np.mean([current_success_rate - rate for rate in self.previous_success_rates[-momentum_window:]])
        avg_path_length_change = np.mean([current_avg_path_length - length for length in self.previous_avg_path_lengths[-momentum_window:]])

        # Non-linear transformation (e.g., hyperbolic tangent)
        transformed_success_rate_change = np.tanh(avg_success_rate_change)
        transformed_path_length_change = np.tanh(avg_path_length_change)

        # Adjust alpha (Pheromone Importance) with non-linear scaling and diversity factor
        self.alpha *= (1.05 + (0.05 * transformed_success_rate_change)) * diversity_factor

        # Adjust beta (Heuristic Importance) with non-linear scaling and inverse diversity factor
        self.beta *= (1.05 + (0.05 * transformed_path_length_change)) / diversity_factor

        # Adjust decay rate with normalization, asymptotic adjustment, and diversity factor
        decay_adjustment = np.tanh(current_avg_path_length - (self.previous_avg_path_lengths[-1] if self.previous_avg_path_lengths else current_avg_path_length))
        self.decay *= (1 + decay_adjustment) * diversity_factor
        self.decay = min(0.5, max(0.01, self.decay))

        # Ensure alpha and beta are within reasonable bounds
        self.alpha = max(0.1, min(self.alpha, 5))
        self.beta = max(0.1, min(self.beta, 5))

        # Update previous values for next iteration
        self.previous_success_rates.append(current_success_rate)
        self.previous_avg_path_lengths.append(current_avg_path_length)
        if len(self.previous_success_rates) > momentum_window:
            self.previous_success_rates.pop(0)
            self.previous_avg_path_lengths.pop(0)

        # Logging for monitoring
        print(f"Iteration: {iteration}, Alpha: {self.alpha:.4f}, Beta: {self.beta:.4f}, Decay: {self.decay:.4f}, Diversity: {diversity_factor:.4f}")

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

            # Dynamic local pheromone update as the ant moves
            self.graph.add_pheromone(current, next_node, self.local_evaporation, self.local_evaporation)

            # Dynamic global evaporation
            self.graph.evaporate_edge_pheromone(current, next_node, self.decay)

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
        pheromones = np.array([max(min(self.graph.pheromone_level(current, neighbor), 10), 1e-5) for neighbor in neighbors])
        heuristic = np.array([max(min(1.0 / (self.graph.distance(neighbor, goal) + 1e-5), 10), 1e-5) for neighbor in neighbors])

        probabilities = pheromones ** self.alpha * heuristic ** self.beta
        prob_sum = probabilities.sum()

        if prob_sum > 0 and not np.isnan(probabilities).any() and not np.isinf(probabilities).any():
            probabilities /= prob_sum
        else:
            probabilities = np.full(len(neighbors), 1.0 / len(neighbors))

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
grid_size = (20, 20)
obstacles = [(x, y) for x in range(8, 15) for y in range(8, 15)]  # Example obstacle
graph = Graph(grid_size, obstacles)
ant_colony = AntColony(graph, n_ants=10, n_iterations=20, decay=0.1, local_evaporation=0.01, alpha=1, beta=1)

start = (0, 0)
goal = (18, 18)
path, length = ant_colony.run(start, goal)
visualize_path(grid_size, obstacles, path)