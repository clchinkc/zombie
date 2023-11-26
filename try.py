import heapq

import matplotlib.pyplot as plt
import numpy as np


class Node:
    def __init__(self, position):
        self.position = np.array(position)
        self.neighbors = []
        self.g = float('inf')  # Cost to start node
        self.rhs = float('inf')  # One-step lookahead cost
        self.parent = None

    def distance(self, other):
        return np.linalg.norm(self.position - other.position)

    def __lt__(self, other):  # This will help with priority queue
        return (self.rhs, self.g) < (other.rhs, other.g)

class Graph:
    def __init__(self):
        self.nodes = []
        self.queue = []
        self.start = None
        self.goal = None

    def add_node(self, position):
        node = Node(position)
        self.nodes.append(node)
        return node

    def add_edge(self, node1, node2):
        cost = node1.distance(node2)
        node1.neighbors.append((node2, cost))
        node2.neighbors.append((node1, cost))

    def remove_node(self, node):
        self.nodes.remove(node)
        for neighbor, _ in node.neighbors:
            neighbor.neighbors = [(n, c) for n, c in neighbor.neighbors if n != node]

    def update_vertex(self, node):
        if node != self.goal:
            # Find the neighbor with the minimum g + cost value
            min_neighbor = min(node.neighbors, key=lambda n: n[0].g + n[1], default=(None, float('inf')))
            node.rhs = min_neighbor[1] + min_neighbor[0].g if min_neighbor[0] is not None else float('inf')
            node.parent = min_neighbor[0]  # Update parent to the node that provides the best path
            
        if node in self.queue:
            self.queue.remove(node)  # Remove the node from the queue to update its priority
            heapq.heapify(self.queue)
        if node.g != node.rhs:
            heapq.heappush(self.queue, node)  # Push it back with the updated values


    def compute_shortest_path(self):
        while self.queue and (self.start.rhs != self.start.g or 
                            any(node.g > node.rhs for node in self.queue)):
            current = heapq.heappop(self.queue)
            if current.g > current.rhs:
                current.g = current.rhs
                for neighbor, cost in current.neighbors:
                    if neighbor.g > current.g + cost:  # Check if a better path is found
                        neighbor.g = current.g + cost
                        neighbor.parent = current  # Update parent pointer
                        self.update_vertex(neighbor)
            else:
                current.g = float('inf')
                for neighbor, cost in current.neighbors:
                    if neighbor.parent == current:  # If the current node is the parent
                        neighbor.rhs = min(neighbor.rhs, cost + self.compute_rhs(neighbor))
                    self.update_vertex(neighbor)
                self.update_vertex(current)

    def compute_rhs(self, node):
        # This should return the rhs value based on the neighbors' g-values and costs.
        # If the node is the goal, the rhs is by definition 0.
        if node == self.goal:
            return 0
        return min(neighbor.g + cost for neighbor, cost in node.neighbors)


    def initialize(self, start, goal):
        self.start = start
        self.goal = goal
        self.start.g = float('inf')
        self.start.rhs = 0
        self.goal.parent = None
        heapq.heappush(self.queue, self.start)
        self.compute_shortest_path()

    def get_path(self):
        # Reconstruct the path from goal to start
        path = []
        current = self.goal
        while current is not None:
            path.append(current)
            current = current.parent

        # Reverse the path so it goes from start to goal
        return path[::-1]

def visualize_graph(graph, start, goal, path=None, obstacle_nodes=None):
    plt.figure(figsize=(10, 10))
    for node in graph.nodes:
        for neighbor, _ in node.neighbors:
            plt.plot([node.position[0], neighbor.position[0]], [node.position[1], neighbor.position[1]], 'k-', lw=0.5)
    plt.scatter([node.position[0] for node in graph.nodes if node not in obstacle_nodes], 
                [node.position[1] for node in graph.nodes if node not in obstacle_nodes], c='blue')
    if obstacle_nodes:
        plt.scatter([node.position[0] for node in obstacle_nodes], 
                    [node.position[1] for node in obstacle_nodes], c='black', marker='s', label='Obstacle')
    plt.scatter([start.position[0]], [start.position[1]], c='green', marker='o', label='Start')
    plt.scatter([goal.position[0]], [goal.position[1]], c='red', marker='x', label='Goal')
    if path:
        plt.plot([node.position[0] for node in path], [node.position[1] for node in path], 'r-', lw=2, label='Path')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage of the Graph
graph = Graph()
node_spacing = 1
for x in np.arange(0, 10, node_spacing):
    for y in np.arange(0, 10, node_spacing):
        graph.add_node((x, y))

# Connect nodes in the graph
for node1 in graph.nodes:
    for node2 in graph.nodes:
        if node1 != node2 and node1.distance(node2) <= 1.5 * node_spacing:
            graph.add_edge(node1, node2)

start = graph.nodes[0]  # Start at the first node
goal = graph.nodes[-1]  # Goal is the last node
graph.initialize(start, goal)

# Now add obstacles and update the graph as needed
obstacle_nodes = []
for node in graph.nodes:
    if 3 <= node.position[0] <= 7 and 3 <= node.position[1] <= 7:
        obstacle_nodes.append(node)
        graph.remove_node(node)
graph.initialize(start, goal)

# Visualization
final_path = graph.get_path()
print(f'Final path : {[node.position for node in final_path]}')
visualize_graph(graph, start, goal, final_path, obstacle_nodes)