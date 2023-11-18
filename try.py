import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt


class Graph:
    def __init__(self, grid_size, obstacles):
        self.grid_size = grid_size
        self.obstacles = set(obstacles)
        self.pheromones = {}

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Adjacent neighbors
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and (nx, ny) not in self.obstacles:
                neighbors.append((nx, ny))
        return neighbors

    def pheromone_level(self, node1, node2):
        return self.pheromones.get((node1, node2), 0.1)  # Default pheromone level

    def add_pheromone(self, node1, node2, amount):
        self.pheromones[(node1, node2)] = self.pheromones.get((node1, node2), 0.1) + amount

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

    def run(self, start, goal, visualizer):
        shortest_path = None
        shortest_path_length = float('inf')

        for iteration in range(self.n_iterations):
            paths = self.generate_paths(start, goal)
            
            # Update visualizer
            visualizer.update(iteration, paths, self.graph.pheromones, shortest_path)
            
            current_success_rate = self.calculate_success_rate(paths)
            self.previous_success_rates.append(current_success_rate)
            current_avg_path_length = self.calculate_average_path_length(paths)
            self.previous_avg_path_lengths.append(current_avg_path_length)

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

    def adapt_parameters(self, iteration, current_success_rate, current_avg_path_length, momentum_window=5):
        if iteration <= momentum_window:
            return  # Ensure enough data points for momentum calculation

        # Calculate average changes over the momentum window
        avg_success_rate_change = np.mean([current_success_rate - rate for rate in self.previous_success_rates[-momentum_window:]])
        avg_path_length_change = np.mean([current_avg_path_length - length for length in self.previous_avg_path_lengths[-momentum_window:]])

        # Non-linear transformation (e.g., hyperbolic tangent)
        transformed_success_rate_change = np.tanh(avg_success_rate_change)
        transformed_path_length_change = np.tanh(avg_path_length_change)

        # Adjust alpha (Pheromone Importance) with non-linear scaling
        self.alpha *= (1.05 + (0.05 * transformed_success_rate_change))

        # Adjust beta (Heuristic Importance) with non-linear scaling
        self.beta *= (1.05 + (0.05 * transformed_path_length_change))

        # Ensure alpha and beta are within reasonable bounds
        self.alpha = max(0.1, min(self.alpha, 5))
        self.beta = max(0.1, min(self.beta, 5))

        # Adjust decay rate with normalization, and asymptotic adjustment
        decay_adjustment = np.tanh(current_avg_path_length - (self.previous_avg_path_lengths[-1] if self.previous_avg_path_lengths else current_avg_path_length))
        self.decay *= (1.05 + (0.05 * decay_adjustment))
        self.decay = min(0.1, max(0.001, self.decay))

        # Update previous values for next iteration
        self.previous_success_rates.append(current_success_rate)
        self.previous_avg_path_lengths.append(current_avg_path_length)
        if len(self.previous_success_rates) > momentum_window:
            self.previous_success_rates.pop(0)
            self.previous_avg_path_lengths.pop(0)

        # Logging for monitoring
        print(f"Iteration: {iteration}, Alpha: {self.alpha:.4f}, Beta: {self.beta:.4f}, Decay: {self.decay:.4f}")

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

            # Local pheromone update
            pheromone_amount = 1.0 / len(path)  # Example: using the number of steps as a simpler metric
            self.graph.add_pheromone(current, next_node, pheromone_amount)

            # Proceed to the next node
            current = next_node

            # Break if path becomes excessively long (to avoid infinite loops)
            if len(path) > self.graph.grid_size[0] * self.graph.grid_size[1]:
                break

        # Pheromone evaporation and length calculation after path completion
        self.graph.evaporate_pheromones(self.decay)
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

    def path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.graph.distance(path[i], path[i + 1])
        return length


class AntColonyVisualizer:
    def __init__(self, grid_size, obstacles):
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.iteration_paths = []
        self.pheromone_levels = []
        self.shortest_paths = []

    def update(self, iteration, paths, pheromone_map, shortest_path):
        self.iteration_paths.append((iteration, paths))
        self.pheromone_levels.append(pheromone_map.copy())
        self.shortest_paths.append(shortest_path)

    def visualize_pheromone_evolution(self):
        # Animated visualization of pheromone intensity over iterations
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])

        # Initial setup for the heatmap
        pheromone_grid = np.zeros(self.grid_size)
        heatmap = ax.imshow(pheromone_grid.T, cmap='hot', origin='lower')
        colorbar = fig.colorbar(heatmap, ax=ax, orientation='vertical')

        def animate(i):
            ax.clear()
            # Update the pheromone grid
            pheromone_grid = np.zeros(self.grid_size)
            for (node1, node2), intensity in self.pheromone_levels[i].items():
                x, y = (np.array(node1) + np.array(node2)) / 2
                pheromone_grid[int(x), int(y)] = intensity
            
            # Update the heatmap
            heatmap.set_data(pheromone_grid.T)
            ax.imshow(pheromone_grid.T, cmap='hot', origin='lower')
            ax.set_title(f"Iteration: {i}")

        ani = animation.FuncAnimation(fig, animate, frames=len(self.iteration_paths), interval=1000)
        plt.show()

    def visualize_path_evolution(self):
        # Animated visualization of path evolution over iterations, highlighting the shortest path
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])

        def animate(i):
            ax.clear()
            iteration, paths = self.iteration_paths[i]
            shortest_path = self.shortest_paths[i]
            # Plot all paths
            for path, _ in paths:
                ax.plot(*zip(*path), marker='o', color='gray', alpha=0.5)
            # Highlight the shortest path
            if shortest_path:
                ax.plot(*zip(*shortest_path), marker='o', color='red', linewidth=2)
            ax.set_title(f"Iteration: {iteration}")

        ani = animation.FuncAnimation(fig, animate, frames=len(self.iteration_paths), interval=1000)
        plt.show()

# Example Usage
grid_size = (20, 20)
obstacles = [(x, y) for x in range(8, 15) for y in range(8, 15)]  # Example obstacle
graph = Graph(grid_size, obstacles)
ant_colony = AntColony(graph, n_ants=5, n_iterations=10, decay=0.0001, local_evaporation=0.001, alpha=1, beta=1)

visualizer = AntColonyVisualizer(grid_size, obstacles)

start = (0, 0)
goal = (18, 18)
path, length = ant_colony.run(start, goal, visualizer)

# Visualization
visualizer.visualize_pheromone_evolution()
visualizer.visualize_path_evolution()