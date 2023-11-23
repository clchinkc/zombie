import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon


def is_collision(p1, p2, obstacles, buffer=1.0):
    line = LineString([p1, p2])
    buffered_line = line.buffer(buffer)
    return any(buffered_line.intersects(obstacle) for obstacle in obstacles)

def nearest_node(node_list, random_point):
    random_node = TreeNode(Point(random_point))
    return min(node_list, key=lambda node: node.distance(random_node))

def find_nearby_nodes(tree, new_node, radius):
    return [node for node in tree if node.distance(new_node) <= radius]

def steer(from_node, to_point, step_size, obstacles):
    direction = np.array(to_point) - np.array((from_node.point.x, from_node.point.y))
    length = np.linalg.norm(direction)
    direction = direction / length if length != 0 else direction

    new_point = np.array((from_node.point.x, from_node.point.y)) + step_size * direction
    if not is_collision(from_node.point, Point(new_point), obstacles):
        return new_point
    else:
        return None


def dynamic_step_size(from_node, to_point, step_size, goal, obstacles):
    goal_distance = float(np.linalg.norm(np.array(to_point) - np.array(goal)))
    step = min(step_size, goal_distance)

    obstacle_distances = [obstacle.distance(Point(from_node.point)) for obstacle in obstacles]
    if obstacle_distances:
        min_obstacle_distance = min(obstacle_distances)
        # If close to an obstacle, reduce step size
        if min_obstacle_distance < step_size:
            step = max(step, min_obstacle_distance / 2)  # Ensure a minimum step
    return step


def random_sampling(space_size, obstacles):
    while True:
        x = np.random.uniform(0, space_size[0])
        y = np.random.uniform(0, space_size[1])
        point = Point(x, y)
        if not any(obstacle.contains(Point(point)) for obstacle in obstacles):
            return x, y

def goal_biased_sampling(space_size, goal, obstacles, alpha=0.5):
    if np.random.rand() > alpha:
        return random_sampling(space_size, obstacles)
    else:
        std_dev = min(space_size) / 6
        # Sample a point from a gaussian distribution centered at the goal
        while True:
            point = np.random.normal(goal, std_dev)
            if 0 <= point[0] <= space_size[0] and 0 <= point[1] <= space_size[1] and not any(obstacle.contains(Point(point)) for obstacle in obstacles):
                return point

def bridge_sampling(space_size, obstacles, bridge_length=5):
    while True:
        midpoint = random_sampling(space_size, obstacles)
        direction = np.random.uniform(-1, 1, size=2)
        direction /= np.linalg.norm(direction)
        point_a = midpoint + bridge_length * direction / 2
        point_b = midpoint - bridge_length * direction / 2
        if not is_collision(Point(point_a), Point(point_b), obstacles) and 0 <= point_a[0] <= space_size[0] and 0 <= point_a[1] <= space_size[1] and 0 <= point_b[0] <= space_size[0] and 0 <= point_b[1] <= space_size[1]:
            return midpoint

def gaussian_sampling(tree, start, goal, space_size, obstacles, alpha=0.5, std_dev=None):
    if std_dev is None:
        std_dev = min(space_size) / 8  # Standard deviation can be adjusted

    while True:
        # Choose a center point from the current node
        if np.random.rand() > alpha:
            center = np.array((tree[-1].point.x, tree[-1].point.y))
        else:
            # Choose a center point from a random node
            node = np.random.choice(tree)
            center = np.array((node.point.x, node.point.y))

        # Sample from Gaussian distribution centered around the chosen point
        x, y = np.random.normal(center, std_dev, size=2)
        point = Point(x, y)

        # Check if the sampled point is valid
        if 0 <= x <= space_size[0] and 0 <= y <= space_size[1] and not any(obstacle.contains(point) for obstacle in obstacles):
            return (x, y)

def obstacle_sampling(space_size, obstacles, buffer=20.0, min_distance=1.0, alpha=0.5):
    if np.random.rand() > alpha:
        return random_sampling(space_size, obstacles)
    else:
        while True:
            # Randomly select an obstacle
            obstacle = np.random.choice(obstacles)
            obstacle_center = obstacle.centroid.coords[0]

            # Sample a random angle and distance
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(min_distance, buffer)
            
            # Calculate x and y coordinates
            x = obstacle_center[0] + distance * np.cos(angle)
            y = obstacle_center[1] + distance * np.sin(angle)
            point = Point(x, y)

            # Check if the point is valid (within space bounds and not inside the obstacle)
            if 0 <= point.x <= space_size[0] and 0 <= point.y <= space_size[1] and not any(obstacle.contains(Point(point)) for obstacle in obstacles):
                return x, y

def dynamic_domain_sampling(tree, space_size, obstacles, expand_radius=10.0):
    # Focus on leaf nodes
    leaf_nodes = [node for node in tree if node.parent is not None and node not in tree[:-1]]

    if not leaf_nodes:
        # Fallback to the entire tree if no leaf nodes are found
        leaf_nodes = tree

    # Choose a random leaf node
    random_node = np.random.choice(leaf_nodes)

    while True:
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, expand_radius)
        x = random_node.point.x + distance * np.cos(angle)
        y = random_node.point.y + distance * np.sin(angle)

        point = Point(x, y)
        if 0 <= x <= space_size[0] and 0 <= y <= space_size[1] and not any(obstacle.contains(point) for obstacle in obstacles):
            return (x, y)


class TreeNode:
    def __init__(self, point, parent=None):
        self.point = point
        self.parent = parent

    def distance(self, other):
        return self.point.distance(other.point)

    def __repr__(self):
        return f"TreeNode({self.point.x}, {self.point.y})"

def tree_distance(node, start_node):
    distance = 0
    current_node = node
    while current_node != start_node:
        distance += current_node.distance(current_node.parent)
        current_node = current_node.parent
    return distance


def ellipse_sampling(space_size, start, goal, c_best, obstacles):
    center = (np.array(start) + np.array(goal)) / 2
    axis_length = c_best / 2
    while True:
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, axis_length)
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        if 0 <= x <= space_size[0] and 0 <= y <= space_size[1] and not any(obstacle.contains(Point(x, y)) for obstacle in obstacles):
            return x, y


def rrt(start, goal, obstacles, num_iterations, step_size, space_size):
    start_node = TreeNode(Point(start))
    tree = [start_node]
    c_best = float('inf')
    goal_reached = False
    goal_node = None

    for i in range(num_iterations):
        if goal_reached:
            random_point = ellipse_sampling(space_size, start, goal, c_best, obstacles)
        elif np.random.rand() < 0.1:  # 10% chance to sample the goal
            random_point = goal
        else:
            # Choose sampling method based on user input
            if sampling_method == 'goal':
                random_point = goal_biased_sampling(space_size, goal, obstacles)
            elif sampling_method == 'bridge':
                random_point = bridge_sampling(space_size, obstacles)
            elif sampling_method == 'gaussian':
                random_point = gaussian_sampling(tree, start, goal, space_size, obstacles)
            elif sampling_method == 'obstacle':
                random_point = obstacle_sampling(space_size, obstacles)
            elif sampling_method == 'dynamic_domain':
                random_point = dynamic_domain_sampling(tree, space_size, obstacles)
            else:
                random_point = random_sampling(space_size, obstacles)

        nearest = nearest_node(tree, random_point)
        dynamic_size = dynamic_step_size(nearest, random_point, step_size, goal, obstacles)
        new_point_coords = steer(nearest, random_point, dynamic_size, obstacles)
        if new_point_coords is not None:
            new_node = TreeNode(Point(new_point_coords), nearest)
            tree.append(new_node)
            
            # RRT* Rewiring Logic
            gamma = 2 * (space_size[0] * space_size[1]) / np.pi  # Constant
            search_radius = min(30.0, (gamma * (np.log(len(tree)) / len(tree)) ** 0.5))
            nearby_nodes = find_nearby_nodes(tree, new_node, search_radius)
            for node in nearby_nodes:
                if node == nearest or node == new_node.parent:
                    continue
                if tree_distance(new_node, start_node) + new_node.distance(node) < tree_distance(node, start_node):
                    if not is_collision(new_node.point, node.point, obstacles):
                        node.parent = new_node
                        # Update c_best if a new node is added to the tree
                        if tree_distance(node, start_node) < c_best:
                            c_best = tree_distance(node, start_node)

            # Check if the goal is reached and update c_best
            if Point(new_point_coords).distance(Point(goal)) <= dynamic_size:
                
                if not goal_reached:
                    print("Reached the goal at iteration", i)
                
                goal_reached = True
                goal_node = new_node
                c_best = tree_distance(goal_node, start_node)

    if not goal_reached:
        print("Goal not reached!")

    return tree, goal_reached, goal_node

def find_path_to_goal(goal_node, start_node):
    if goal_node is None:
        return None

    path = []
    current_node = goal_node
    while current_node != start_node:
        path.append(current_node.point)
        current_node = current_node.parent
    path.append(start_node.point)  # Include the start node

    return path[::-1]  # Return reversed path starting from the beginning


def smooth_path(path, obstacles, max_iterations=50):
    if len(path) < 3:  # No smoothing needed for paths with less than 3 points
        return path

    smooth_path = path.copy()
    iteration = 0

    while iteration < max_iterations:
        changed = False

        i = 0
        while i < len(smooth_path) - 2:
            max_j = i + 1  # Initialize max_j as the next immediate point

            for j in range(i + 2, len(smooth_path)):
                if not is_collision(smooth_path[i], smooth_path[j], obstacles):
                    max_j = j  # Find the furthest point that can be directly reached without collision

            if max_j != i + 1:
                # Remove intermediate points between i and max_j
                smooth_path[i + 1:max_j] = [smooth_path[max_j]]
                changed = True

            i = max_j  # Move to the next point that needs checking

        iteration += 1

        if not changed:
            break  # Exit early if no changes are made in the iteration

    return smooth_path




def plot_obstacles(obstacles, ax):
    for obstacle in obstacles:
        x, y = obstacle.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='orange', ec='none')

def plot_tree(tree, ax):
    for point in tree:
        if point.parent is not None:
            ax.plot([point.point.x, point.parent.point.x], [point.point.y, point.parent.point.y], 'blue')

def plot_points(tree, start, goal, ax):
    for point in tree:
        ax.plot(point.point.x, point.point.y, 'bo')
    ax.plot(start[0], start[1], 'go', label='Start')
    ax.plot(goal[0], goal[1], 'ro', label='Goal')

def plot_path(path, color, line_style, label, linewidth, ax):
    for i in range(len(path)-1):
        if i == 0:
            # Add label only for the first segment
            ax.plot([path[i].x, path[i+1].x], [path[i].y, path[i+1].y], color=color, linestyle=line_style, linewidth=linewidth, label=label)
        else:
            # Plot without label for the rest of the segments
            ax.plot([path[i].x, path[i+1].x], [path[i].y, path[i+1].y], color=color, linestyle=line_style, linewidth=linewidth)




# Example usage:
space_size = (100, 100)
start = (0, 0)
goal = (90, 90)
obstacles = [Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
            Polygon([(50, 50), (60, 50), (60, 60), (50, 60)]),
            Polygon([(70, 70), (80, 70), (80, 80), (70, 80)])]
num_iterations = 500
step_size = 5.

# Choose the sampling method: 'random', 'goal', 'bridge', 'gaussian', 'obstacle', or 'dynamic_domain'
sampling_method = 'gaussian'

# Run RRT
path, goal_reached, goal_node = rrt(start, goal, obstacles, num_iterations, step_size, space_size)

# Find the correct path
correct_path = find_path_to_goal(goal_node, path[0])

# Find the smooth path
smoothed_path = smooth_path(correct_path, obstacles)

# Plotting the result
fig, ax = plt.subplots()
ax.set_xlim(0, space_size[0])
ax.set_ylim(0, space_size[1])

# Plot elements
plot_obstacles(obstacles, ax)
plot_tree(path, ax)
plot_points(path, start, goal, ax)
if correct_path:
    plot_path(correct_path, 'r', '-', 'Correct Path', 3, ax)
if smoothed_path:
    plot_path(smoothed_path, 'g', '-', 'Smooth Path', 3, ax)

plt.legend(loc='upper left')
plt.title('RRT Path Planning')
plt.show()
