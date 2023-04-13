"""
There are many ways to implement more complex algorithms for controlling the movements of people and zombies in a simulation. Here are a few examples of more advanced approaches you could consider:

Pathfinding algorithms: You could use algorithms such as A* or Dijkstra's algorithm to allow people and zombies to intelligently navigate the school layout and avoid obstacles. This could be useful for simulating more realistic or strategic movements, such as people trying to find the best route to escape the school or zombies trying to track down nearby survivors.

Flocking algorithms: You could use algorithms such as Boids or Reynolds' steering behaviors to simulate the collective movements of groups of people or zombies. This could be used to model the behavior of crowds or hordes, and could help to create more realistic and dynamic simulations.

Machine learning algorithms: You could use machine learning techniques such as reinforcement learning or decision trees to train people and zombies to make more intelligent decisions about their movements. This could allow the simulation to adapt and improve over time, and could potentially enable the people and zombies to learn from their experiences and make more effective decisions in the future.
"""
import heapq
import queue
import random

"""
maze = [
    ["#", "O", "#", "#", "#", "#", "#", "#", "#"],
    ["#", " ", " ", " ", " ", " ", " ", " ", "#"],
    ["#", " ", "#", "#", " ", "#", "#", " ", "#"],
    ["#", " ", "#", " ", " ", " ", "#", " ", "#"],
    ["#", " ", "#", " ", "#", " ", "#", " ", "#"],
    ["#", " ", "#", " ", "#", " ", "#", " ", "#"],
    ["#", " ", "#", " ", "#", " ", "#", "#", "#"],
    ["#", " ", " ", " ", " ", " ", " ", " ", "#"],
    ["#", "#", "#", "#", "#", "#", "#", "X", "#"]
]
"""

# generates a maze of the given size with natural path
# but does not ensure that the maze is solvable
def generate_maze(size):
    maze = [["#" for _ in range(size)] for _ in range(size)]
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if i == 0 or i == len(maze) - 1 or j == 0 or j == len(maze[0]) - 1:
                maze[i][j] = "@"
    if random.randint(0, 1) == 0:
        start = (0, random.randint(1, size-2))
        exit = (size-1, random.randint(1, size-2))
    else:
        start = (random.randint(1, size-2), 0)
        exit = (random.randint(1, size-2), size-1)
    
    natural_rendering(size, maze, start, exit)
    
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if i == 0 or i == len(maze) - 1 or j == 0 or j == len(maze[0]) - 1:
                maze[i][j] = "#"
    maze[start[0]][start[1]] = "O"
    maze[exit[0]][exit[1]] = "X"

    return maze, start, exit

def natural_rendering(size, maze, start, exit):
    back = [start]
    pos = start
    while back:
        if pos == exit:
            break
        choices = []
        if size-2 >= pos[0] >= 1 and maze[pos[0]-2][pos[1]] == "#":
            choices.append([pos[0]-2, pos[1]])
        if 0 <= pos[0] <= size-3 and maze[pos[0]+2][pos[1]] == "#":
            choices.append([pos[0]+2, pos[1]])
        if size-2 >= pos[1] >= 1 and maze[pos[0]][pos[1]-2] == "#":
            choices.append([pos[0], pos[1]-2])
        if 0 <= pos[1] <= size-3 and maze[pos[0]][pos[1]+2] == "#":
            choices.append([pos[0], pos[1]+2])
        if choices:
            choice = random.choice(choices)
            maze[(pos[0]+choice[0])//2][(pos[1]+choice[1])//2] = " "
            maze[choice[0]][choice[1]] = " "
            pos = choice
            back.append(pos)
        else:
            pos = back.pop(random.randint(0, len(back)-1))

# may use depth first search
# https://makeschool.org/mediabook/oa/tutorials/trees-and-mazes/generating-a-maze-with-dfs/
# or wave function collapse
# https://github.com/avihuxp/WaveFunctionCollapse
# https://www.youtube.com/watch?v=2SuvO4Gi7uY
# https://youtu.be/rI_y2GAlQFM
# https://youtu.be/TlLIOgWYVpI
# https://youtu.be/20KHNA9jTsE
# https://youtu.be/TO0Tx3w5abQ

def print_maze(maze):
    for row in maze:
        for cell in row:
            print(cell, end=" ")
        print()
    print()

# use the location of the person that find the path
def start_location(maze):
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == "O":
                return row, col

# use the location of the destination
def end_location(maze):
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == "X":
                return row, col


def find_neighbors(maze, row, col):
    """
    Find the valid neighbors of a given cell.
    """
    
    neighbors = []

    if row > 0:  # UP
        neighbors.append((row - 1, col))
    if row + 1 < len(maze):  # DOWN
        neighbors.append((row + 1, col))
    if col > 0:  # LEFT
        neighbors.append((row, col - 1))
    if col + 1 < len(maze[0]):  # RIGHT
        neighbors.append((row, col + 1))

    return neighbors

# find path using A* algorithm
def find_path(maze, start, end):
    # Find the shortest path from start to end using the A* algorithm.

    start_pos = start
    end_pos = end

    # priority queue for storing unexplored nodes
    heap = []
    heapq.heappush(heap, (0, start_pos, []))

    visited = set()  # set of visited nodes

    while heap:
        cost, current_pos, path = heapq.heappop(heap)
        row, col = current_pos

        if current_pos == end_pos:
            return path

        if current_pos in visited:
            continue

        neighbors = find_neighbors(maze, row, col)
        for neighbor in neighbors:
            if neighbor in visited:
                continue

            r, c = neighbor
            if maze[r][c] == "#":
                continue

            # calculate cost of reaching this neighbor
            g = cost + 1  # movement cost
            h = manhattan_distance(
                neighbor, end_pos)  # heuristic cost
            f = g + h  # total cost

            new_path = path + [neighbor]
            heapq.heappush(heap, (f, neighbor, new_path))
            visited.add(current_pos)

    return path

def manhattan_distance(start, end):
    x1, y1 = start
    x2, y2 = end
    return abs(x1 - x2) + abs(y1 - y2)

"""
def find_path(maze, start, end):

    # Find the shortest path from start to end using the A* algorithm.

    start_pos = start
    end_pos = end

    q = queue.Queue()
    q.put((start_pos, [start_pos]))

    visited = set()

    while not q.empty():
        current_pos, path = q.get()
        row, col = current_pos

        if maze[row][col] == end_pos:
            return path

        neighbors = find_neighbors(maze, row, col)
        for neighbor in neighbors:
            if neighbor in visited:
                continue

            r, c = neighbor
            if maze[r][c] == "#":
                continue

            new_path = path + [neighbor]
            q.put((neighbor, new_path))
            visited.add(neighbor)

    return new_path
"""

# may return first element of the path for the destination of this epoch

def print_new_maze(maze, path, start, end):
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if (row, col) == start:
                print("O", end=" ")
            elif (row, col) == end:
                print("X", end=" ")
            elif (row, col) in path:
                print("o", end=" ")
            else:
                print(maze[row][col], end=" ")
        print()
    print()


def path_finding():
    maze, start, exit = generate_maze(10)
    #start = start_location(maze)
    #exit = end_location(maze)
    path = find_path(maze, start, exit)
    print_maze(maze)
    print_new_maze(maze, path, start, exit)


path_finding()

"""
class maze:
	def __init__(self,layout,exits=None,name='Maze',time=0,random=None,end=None):
		self.layout=layout # a list of lists of numbers
		self.exits=exits # a list of the exits in the order of north, east, south, west
		self.name=name.capitalize() # the name of the maze
		self.time=time*60
		self.random=random # the size of the random maze
		self.end=end # the end message
	def get_maze(self):
		if self.random:
			return generate(self.random*4-1,self.up)
		else:
			return self.layout
	def get_exits(self):
		return self.exits

# master_maze a list of all the mazes in the game and some places storing special items
"""
"""
from heapq import heappop, heappush
from math import sqrt

def plan_route(start, end, prob_map):
    # Initialize the open and closed sets
    open_set = [(0, start)] # priority queue with the start node as the first element
    closed_set = set()
    
    # Initialize the cost and parent dictionaries
    cost = {start: 0}
    parent = {start: None}
    
    # Define the heuristic function
    def heuristic(node):
        dx = end[0] - node[0]
        dy = end[1] - node[1]
        return sqrt(dx**2 + dy**2)
    
    # Define the edge cost function
    def edge_cost(node1, node2):
        prob = prob_map[node2] # probability of encountering zombies
        if prob == 0:
            return float('inf') # avoid impassable nodes
        return 1/prob # inverse probability as edge cost
    
    # Start the search
    while open_set:
        # Get the node with the lowest cost from the priority queue
        current_cost, current_node = heappop(open_set)
        if current_node == end:
            # Found the goal, reconstruct the path
            path = [current_node]
            while parent[current_node]:
                path.append(parent[current_node])
                current_node = parent[current_node]
            path.reverse()
            return path
        
        # Add the current node to the closed set
        closed_set.add(current_node)
        
        # Expand the neighbors of the current node
        for neighbor in get_neighbors(current_node):
            if neighbor in closed_set:
                continue
            
            # Calculate the tentative cost of reaching the neighbor
            tentative_cost = cost[current_node] + edge_cost(current_node, neighbor)
            
            if neighbor not in cost or tentative_cost < cost[neighbor]:
                # Update the cost and parent of the neighbor
                cost[neighbor] = tentative_cost
                parent[neighbor] = current_node
                
                # Add the neighbor to the priority queue with the new cost
                priority = tentative_cost + heuristic(neighbor)
                heappush(open_set, (priority, neighbor))
    
    # Goal not found, return None
    return None

"""


# https://realpython.com/python-maze-solver/