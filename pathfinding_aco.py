import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


class Graph:
    def __init__(self, grid_size, obstacles):
        self.grid_size = grid_size
        self.obstacles = set(obstacles)
        self.nodes = self.generate_nodes()
        self.edges = self.generate_edges()

    def generate_nodes(self):
        nodes = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if (i, j) not in self.obstacles:
                    nodes.append((i, j))
        return nodes

    def generate_edges(self):
        edges = {}
        for node in self.nodes:
            edges[node] = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (node[0] + dx, node[1] + dy)
                if neighbor in self.nodes:
                    edges[node].append(neighbor)
        return edges

class Ant:
    def __init__(self, start_node):
        self.current_node = start_node
        self.path = [start_node]
        self.path_length = 0

    def move_to_node(self, graph, next_node, pheromone_map, local_evaporation, goal):
        self.path_length += np.linalg.norm(np.array(next_node) - np.array(self.current_node))
        self.current_node = next_node
        self.path.append(next_node)

        # Dynamic pheromone update
        distance_to_goal = np.linalg.norm(np.array(goal) - np.array(next_node))
        distance_to_obstacle = min([float(np.linalg.norm(np.array(next_node) - np.array(obstacle))) for obstacle in graph.obstacles])
        pheromone_amount = 1 / (self.path_length + distance_to_goal - distance_to_obstacle)
        edge = (self.path[-2], self.path[-1])
        pheromone_map[edge] += pheromone_amount

        # Local pheromone evaporation
        pheromone_map[edge] *= (1 - local_evaporation)

    def choose_next_node(self, graph, pheromone_map, alpha, beta, goal):
        # Enhanced heuristic: distance to goal combined with obstacle avoidance
        neighbors = graph.edges[self.current_node]
        heuristic = np.array([1.0 / (np.linalg.norm(np.array(neighbor) - np.array(goal)) + 1) for neighbor in neighbors])
        pheromones = np.array([pheromone_map.get((self.current_node, neighbor), 0.1) for neighbor in neighbors])

        probabilities = pheromones ** alpha * heuristic ** beta
        probabilities /= probabilities.sum()

        next_index = np.random.choice(len(neighbors), p=probabilities)
        return neighbors[next_index]





class AntColony:
    def __init__(self, graph, n_ants, n_iterations, alpha=1, beta=1, decay=0.1, local_evaporation=0.05):
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.local_evaporation = local_evaporation
        self.pheromone_map = self.initialize_pheromones(graph, 0.1)

    def run(self, start, goal):
        best_path = list()
        best_path_length = float('inf')
        iteration_data = []

        for i in range(self.n_iterations):
            ants = [Ant(start) for _ in range(self.n_ants)]
            total_path_length = 0
            for ant in ants:
                while ant.current_node != goal:
                    next_node = ant.choose_next_node(self.graph, self.pheromone_map, self.alpha, self.beta, goal)
                    ant.move_to_node(self.graph, next_node, self.pheromone_map, self.local_evaporation, goal)
                total_path_length += ant.path_length
                
                if ant.path_length < best_path_length:
                    best_path = ant.path
                    best_path_length = ant.path_length
            
            # Global pheromone evaporation
            self.update_pheromones(self.pheromone_map, self.decay)

            visualizer.update(i, [ant.path for ant in ants], self.pheromone_map, best_path)

            # Record the average path length and pheromone level for this iteration
            avg_path_length = total_path_length / len(ants)
            pheromone_level = sum(self.pheromone_map.values()) / len(self.pheromone_map)
            iteration_data.append((i, avg_path_length, pheromone_level))

        return best_path, iteration_data

    def initialize_pheromones(self, graph, initial_value):
        pheromone_map = {}
        for node in self.graph.nodes:
            for neighbor in self.graph.edges[node]:
                pheromone_map[(node, neighbor)] = initial_value
        return pheromone_map

    def update_pheromones(self, pheromone_map, decay_rate):
        for edge in pheromone_map:
            pheromone_map[edge] *= (1 - decay_rate)


class AntColonyVisualizer:
    def __init__(self, graph, grid_size, obstacles):
        self.graph = graph
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.iteration_paths = []
        self.pheromone_levels = []
        self.shortest_paths = []

    def update(self, iteration, paths, pheromone_map, shortest_path):
        self.iteration_paths.append((iteration, paths))
        self.pheromone_levels.append(pheromone_map.copy())
        self.shortest_paths.append(shortest_path)

    def normalize_pheromone_grid(self, pheromone_map):
        pheromone_grid = np.zeros(self.grid_size)
        for (node1, node2), intensity in pheromone_map.items():
            x, y = (np.array(node1) + np.array(node2)) / 2
            pheromone_grid[int(x), int(y)] = intensity
        
        # Normalize the pheromone grid
        max_intensity = pheromone_grid.max()
        if max_intensity > 0:
            pheromone_grid /= max_intensity
        
        return pheromone_grid

    def visualize_pheromone_evolution(self):
        # Modified visualization for pheromone intensity over iterations with colorbar
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1, self.grid_size[0])
        ax.set_ylim(-1, self.grid_size[1])
        plt.xticks(np.arange(-0.5, self.grid_size[0], 1))
        plt.yticks(np.arange(-0.5, self.grid_size[1], 1))

        # Initial setup for the heatmap
        pheromone_grid = self.normalize_pheromone_grid(self.pheromone_levels[0])
        heatmap = ax.imshow(pheromone_grid.T, cmap='hot', origin='lower')
        colorbar = fig.colorbar(heatmap, ax=ax, orientation='vertical')

        def animate(i):
            # Update the pheromone grid for the current iteration and normalize
            pheromone_grid = self.normalize_pheromone_grid(self.pheromone_levels[i])

            # Update the heatmap data
            heatmap.set_data(pheromone_grid.T)
            ax.set_title(f"Iteration: {i}")

        ani = animation.FuncAnimation(fig, animate, frames=len(self.pheromone_levels), interval=1000)
        plt.show()

    def visualize_path_evolution(self):
        # Modified visualization for path evolution over iterations
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1, self.grid_size[0])
        ax.set_ylim(-1, self.grid_size[1])
        plt.xticks(np.arange(-0.5, self.grid_size[0], 1))
        plt.yticks(np.arange(-0.5, self.grid_size[1], 1))
        ax.grid(True)

        def animate(i):
            ax.clear()
            iteration, paths = self.iteration_paths[i]
            shortest_path = self.shortest_paths[i]

            for path in paths:
                ax.plot([node[0] for node in path], [node[1] for node in path], color='gray', alpha=0.5, linewidth=2)

            if shortest_path:
                ax.plot([node[0] for node in shortest_path], [node[1] for node in shortest_path], color='red', linewidth=3)

            ax.set_title(f"Iteration: {iteration}")

        ani = animation.FuncAnimation(fig, animate, frames=len(self.iteration_paths), interval=1000)
        plt.show()
        

# Example usage
grid_size = (20, 20)
obstacles = [(5, 5), (5, 6), (5, 7), (5, 8), (5, 9)]

graph = Graph(grid_size, obstacles)
ant_colony = AntColony(graph, n_ants=10, n_iterations=10, alpha=1, beta=1, decay=0.01, local_evaporation=0.001)

visualizer = AntColonyVisualizer(graph, grid_size, obstacles)

start = (0, 0)
goal = (19, 19)
best_path, iteration_data = ant_colony.run(start, goal)

visualizer.visualize_pheromone_evolution()
visualizer.visualize_path_evolution()

print(f"Shortest path: {best_path}")
print(f"Path length: {len(best_path)}")
print(f"Iteration data:")
for iteration, avg_path_length, pheromone_level in iteration_data:
    print(f"Iteration: {iteration}, Avg. path length: {avg_path_length}, Pheromone level: {pheromone_level}")
    
"""
Parameter Tuning: The performance of the algorithm is highly dependent on the parameters like alpha, beta, and decay. Consider implementing a mechanism to automatically adjust these parameters based on the performance in initial runs.

Visualization Enhancements: The current visualization is quite basic. You could enhance the visualization by also showing the movement of ants.

Parallelization: The nature of the ant colony algorithm makes it suitable for parallel execution. If performance is a concern, especially for larger grids, consider implementing a parallel version of your algorithm.

Obstacle Handling: Your current implementation considers static obstacles. You can extend this by allowing dynamic obstacles that change position after a certain number of iterations. This will test the adaptability of the ant colony to changing environments.

Multiple Goals: Extend the model to handle scenarios with multiple goals. This can simulate situations where ants need to find the shortest paths to multiple food sources.

Ant Behavior Variation: Introduce variations in ant behavior. For example, some ants could prioritize exploration while others prioritize exploitation, based on a probability distribution. This can help in balancing exploration and exploitation.

Pheromone Evaporation and Diffusion: Implement a more sophisticated pheromone evaporation mechanism where pheromones not only evaporate over time but also diffuse to adjacent cells. This can help in creating more robust pheromone trails.

Performance Metrics: Introduce more performance metrics such as the total number of steps taken by all ants, the number of iterations until the first optimal path is found, and the convergence rate.

"""