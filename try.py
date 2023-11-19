import numpy as np


class Graph:
    def __init__(self, size, obstacles):
        self.size = size
        self.obstacles = obstacles
        self.nodes = self.generate_nodes()
        self.edges = self.generate_edges()

    def generate_nodes(self):
        nodes = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
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

    def node_to_index(self, node):
        return node[0] * self.size[1] + node[1]

    def index_to_node(self, index):
        return (index // self.size[1], index % self.size[1])

class Ant:
    def __init__(self, start_node):
        self.current_node = start_node
        self.path = [start_node]
        self.path_length = 0

    def move_to_node(self, next_node):
        self.path_length += np.linalg.norm(np.array(next_node) - np.array(self.current_node))
        self.current_node = next_node
        self.path.append(next_node)

    def choose_next_node(self, graph, pheromone_map):
        neighbors = graph.edges[self.current_node]
        neighbor_indices = [graph.node_to_index(neighbor) for neighbor in neighbors]
        pheromone_levels = np.array([pheromone_map[(self.current_node, neighbor)] for neighbor in neighbors])
        
        if pheromone_levels.sum() == 0:
            probabilities = None  # Equal probability for all if no pheromones
        else:
            probabilities = pheromone_levels / pheromone_levels.sum()

        next_index = np.random.choice(neighbor_indices, p=probabilities)
        next_node = graph.index_to_node(next_index)
        return next_node

def initialize_pheromones(graph, initial_value):
    pheromone_map = {}
    for node in graph.nodes:
        for neighbor in graph.edges[node]:
            pheromone_map[(node, neighbor)] = initial_value
    return pheromone_map

def update_pheromones(pheromone_map, ants, decay_rate, pheromone_deposit):
    # Evaporate pheromones
    for edge in pheromone_map:
        pheromone_map[edge] *= (1 - decay_rate)

    # Deposit new pheromones
    for ant in ants:
        for i in range(len(ant.path) - 1):
            edge = (ant.path[i], ant.path[i + 1])
            pheromone_map[edge] += pheromone_deposit / ant.path_length

def aco_algorithm(graph, start, goal, num_ants, num_iterations, decay_rate, pheromone_deposit):
    pheromone_map = initialize_pheromones(graph, initial_value=1.0)
    best_path = None
    best_path_length = float('inf')

    for _ in range(num_iterations):
        ants = [Ant(start) for _ in range(num_ants)]

        for ant in ants:
            while ant.current_node != goal:
                next_node = ant.choose_next_node(graph, pheromone_map)
                ant.move_to_node(next_node)

            # Check if the ant's path is better
            if ant.path_length < best_path_length:
                best_path = ant.path
                best_path_length = ant.path_length

        update_pheromones(pheromone_map, ants, decay_rate, pheromone_deposit)

    return best_path

def visualize_path(graph, path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.xlim(-1, graph.size[0])
    plt.ylim(-1, graph.size[1])
    plt.xticks(np.arange(-0.5, graph.size[0], 1))
    plt.yticks(np.arange(-0.5, graph.size[1], 1))
    plt.grid(True)
    plt.plot([node[0] for node in path], [node[1] for node in path], color='red', linewidth=2)
    plt.plot([node[0] for node in graph.obstacles], [node[1] for node in graph.obstacles], 'sk', markersize=10)
    plt.show()

# Define your space size and obstacles
space_size = (20, 20)
obstacles = [(5, 5), (5, 6), (5, 7), (5, 8), (5, 9)]

# Create graph
graph = Graph(space_size, obstacles)

# Run ACO
start_node = (0, 0)
goal_node = (19, 19)
num_ants = 10
num_iterations = 10
decay_rate = 0.1
pheromone_deposit = 1.0

best_path = aco_algorithm(graph, start_node, goal_node, num_ants, num_iterations, decay_rate, pheromone_deposit)
visualize_path(graph, best_path)
