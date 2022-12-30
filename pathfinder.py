"""
There are many ways to implement more complex algorithms for controlling the movements of people and zombies in a simulation. Here are a few examples of more advanced approaches you could consider:

Pathfinding algorithms: You could use algorithms such as A* or Dijkstra's algorithm to allow people and zombies to intelligently navigate the school layout and avoid obstacles. This could be useful for simulating more realistic or strategic movements, such as people trying to find the best route to escape the school or zombies trying to track down nearby survivors.

Flocking algorithms: You could use algorithms such as Boids or Reynolds' steering behaviors to simulate the collective movements of groups of people or zombies. This could be used to model the behavior of crowds or hordes, and could help to create more realistic and dynamic simulations.

Machine learning algorithms: You could use machine learning techniques such as reinforcement learning or decision trees to train people and zombies to make more intelligent decisions about their movements. This could allow the simulation to adapt and improve over time, and could potentially enable the people and zombies to learn from their experiences and make more effective decisions in the future.
"""

import queue
import heapq

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


# pathfinder without stdscr

def print_maze():
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


def find_path(maze, start, end):
    """
    Find the shortest path from start to end using the A* algorithm.
    """

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

# may return first element of the path for the destination of this epoch

"""
def find_path(self, start, end):
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

        neighbors = find_neighbors(row, col)
        for neighbor in neighbors:
            if neighbor in visited:
                continue

            r, c = neighbor
            if self.layout[r][c] == "#":
                continue

            # calculate cost of reaching this neighbor
            g = cost + 1  # movement cost
            h = self.manhattan_distance(
                neighbor, end_pos)  # heuristic cost
            f = g + h  # total cost

            new_path = path + [neighbor]
            heapq.heappush(heap, (f, neighbor, new_path))
            visited.add(current_pos)

    return path

def manhattan_distance(self, start, end):
    x1, y1 = start
    x2, y2 = end
    return abs(x1 - x2) + abs(y1 - y2)
"""


def print_new_maze(path):
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if (row, col) in path:
                print("o", end=" ")
            else:
                print(maze[row][col], end=" ")
        print()
    print()


def path_finding():
    start = start_location(maze)
    end = end_location(maze)
    path = find_path(maze, start, end)
    print_maze()
    print_new_maze(path)


path_finding()
