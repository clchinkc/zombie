import queue

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


def find_start(maze):
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == "O":
                return row, col


def find_end(maze):
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == "X":
                return row, col


def find_neighbors(maze, row, col):
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
    start = find_start(maze)
    end = find_end(maze)
    path = find_path(maze, start, end)
    print_maze()
    print_new_maze(path)


path_finding()
