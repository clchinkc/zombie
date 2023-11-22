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

    def move_to_node(self, graph, next_node, colony, amount, local_evaporation, goal):
        self.path_length += np.linalg.norm(np.array(next_node) - np.array(self.current_node))
        self.current_node = next_node
        self.path.append(next_node)

        # Dynamic pheromone update
        distance_to_goal = np.linalg.norm(np.array(goal) - np.array(next_node))
        distance_to_obstacle = min([float(np.linalg.norm(np.array(next_node) - np.array(obstacle))) for obstacle in graph.obstacles])
        distance_threshold = 1
        adjusted_distance_to_obstacle = max(0, distance_threshold - distance_to_obstacle)
        
        pheromone_amount = amount / np.sqrt(self.path_length**2 + distance_to_goal**2 + adjusted_distance_to_obstacle**2)
        edge = (self.path[-2], self.path[-1])
        colony.set_pheromone(edge, colony.get_pheromone(edge) + pheromone_amount)

        # Local pheromone evaporation
        colony.set_pheromone(edge, colony.get_pheromone(edge) * (1 - local_evaporation))

    def choose_next_node(self, graph, colony, alpha, beta, goal):
        # Enhanced heuristic: distance to goal combined with obstacle avoidance
        neighbors = graph.edges[self.current_node]
        heuristic = np.array([1.0 / (np.linalg.norm(np.array(neighbor) - np.array(goal)) + 1) for neighbor in neighbors])
        pheromones = np.array([colony.get_pheromone((self.current_node, neighbor)) for neighbor in neighbors])

        probabilities = pheromones ** alpha * heuristic ** beta
        probabilities /= probabilities.sum()

        next_index = np.random.choice(len(neighbors), p=probabilities)
        return neighbors[next_index]


class AntColony:
    def __init__(self, graph, n_ants, n_iterations, alpha=1., beta=1., amount=1., decay=0.1, local_evaporation=0.05):
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.amount = amount
        self.decay = decay
        self.local_evaporation = local_evaporation
        self.pheromone_map = {}

    def get_pheromone(self, edge):
        if edge not in self.pheromone_map:
            self.pheromone_map[edge] = 0.1
        return self.pheromone_map[edge]

    def set_pheromone(self, edge, value):
        self.pheromone_map[edge] = value

    def run(self, start, goal):
        best_path = list()
        best_path_length = float('inf')
        iteration_data = []

        for i in range(self.n_iterations):
            ants = [Ant(start) for _ in range(self.n_ants)]
            total_path_length = 0
            for ant in ants:
                while ant.current_node != goal:
                    next_node = ant.choose_next_node(self.graph, self, self.alpha, self.beta, goal)
                    ant.move_to_node(self.graph, next_node, self, self.amount, self.local_evaporation, goal)
                total_path_length += ant.path_length
                
                if ant.path_length < best_path_length:
                    best_path = ant.path
                    best_path_length = ant.path_length
            
            # Global pheromone evaporation
            self.update_pheromones(self.pheromone_map, self.decay)
            
            # Normalize pheromone levels
            self.normalize_pheromones(self.pheromone_map)

            visualizer.update(i, [ant.path for ant in ants], self.pheromone_map, best_path)

            # Record the average path length and pheromone level for this iteration
            avg_path_length = total_path_length / len(ants)
            pheromone_level = sum(self.pheromone_map.values()) / len(self.pheromone_map)
            iteration_data.append((i, avg_path_length, pheromone_level))

        return best_path, iteration_data

    def update_pheromones(self, pheromone_map, decay_rate):
        for edge in pheromone_map:
            self.set_pheromone(edge, self.get_pheromone(edge) * (1 - decay_rate))

    def normalize_pheromones(self, pheromone_map):
        # Normalize pheromone levels so that the highest level is 1
        max_pheromone = max(pheromone_map.values())
        for edge in pheromone_map:
            pheromone_map[edge] /= max_pheromone

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

    def get_pheromone_grid(self, pheromone_map):
        pheromone_grid = np.zeros(self.grid_size)
        for node in self.graph.nodes:
            for neighbor in self.graph.edges[node]:
                edge = (node, neighbor)
                # Ensure the edge is in the pheromone map
                if edge in pheromone_map:
                    pheromone_level = pheromone_map[edge]
                    pheromone_grid[node[0], node[1]] = pheromone_level
                    
        for obstacle in self.obstacles:
            pheromone_grid[obstacle[0], obstacle[1]] = 0
            
        return pheromone_grid

    def visualize_pheromone_evolution(self):
        # Modified visualization for pheromone intensity over iterations with colorbar
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1, self.grid_size[0])
        ax.set_ylim(-1, self.grid_size[1])
        plt.xticks(np.arange(-0.5, self.grid_size[0], 1))
        plt.yticks(np.arange(-0.5, self.grid_size[1], 1))

        # Initial setup for the heatmap
        pheromone_grid = self.get_pheromone_grid(self.pheromone_levels[0])
        heatmap = ax.imshow(pheromone_grid.T, cmap='hot', origin='lower')
        colorbar = fig.colorbar(heatmap, ax=ax, orientation='vertical')

        def animate(i):
            # Update the pheromone grid for the current iteration
            pheromone_grid = self.get_pheromone_grid(self.pheromone_levels[i])

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
ant_colony = AntColony(graph, n_ants=100, n_iterations=100, alpha=1., beta=1., amount=0.1, decay=0.001, local_evaporation=0.00001)

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