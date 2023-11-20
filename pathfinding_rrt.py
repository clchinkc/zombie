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

    step = step_size
    new_point = np.array((from_node.point.x, from_node.point.y)) + step * direction
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

def tree_distance(tree, node, start_node):
    distance = 0
    current_node = node
    while current_node != start_node:
        distance += current_node.distance(current_node.parent)
        current_node = current_node.parent
    return distance

def rrt(start, goal, obstacles, num_iterations, step_size, sampling_method):
    start_node = TreeNode(Point(start))
    tree = [start_node]
    c_best = float('inf')

    for i in range(num_iterations):
        # Goal biasing: occasionally sample the goal
        if np.random.rand() < 0.1:  # 10% chance to sample the goal
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
                if tree_distance(tree, new_node, start_node) + new_node.distance(node) < tree_distance(tree, node, start_node):
                    if not is_collision(new_node.point, node.point, obstacles):
                        node.parent = new_node
                        # Update c_best if a new node is added to the tree
                        if tree_distance(tree, node, start_node) < c_best:
                            c_best = tree_distance(tree, node, start_node)
            
            if Point(new_point_coords).distance(Point(goal)) <= dynamic_size:
                goal_reached = True
                print("Reached the goal at iteration", i)
                return tree, goal_reached  # Return the tree immediately after reaching the goal
    print("Failed to reach the goal.")
    goal_reached = False
    return tree, goal_reached

def smooth_path(path, obstacles, max_iterations=50, tolerance=0.01):
    smooth_path = path.copy()
    iteration = 0
    while iteration < max_iterations:
        changed = False
        for i in range(len(smooth_path) - 2):
            for j in range(len(smooth_path) - 1, i+1, -1):
                if not is_collision(smooth_path[i], smooth_path[j], obstacles):
                    # Update the path by removing intermediate points
                    smooth_path = smooth_path[:i+1] + smooth_path[j:]
                    changed = True
                    break
            if changed:
                break
        iteration += 1
        if not changed:
            break
    return smooth_path



# Example usage:
space_size = (100, 100)
start = (0, 0)
goal = (90, 90)
obstacles = [Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
            Polygon([(50, 50), (60, 50), (60, 60), (50, 60)]),
            Polygon([(70, 70), (80, 70), (80, 80), (70, 80)])]
num_iterations = 1000
step_size = 5.

# Choose the sampling method: 'random', 'goal', 'bridge', 'gaussian', 'obstacle', or 'dynamic_domain'
sampling_method = 'dynamic_domain'

path, goal_reached = rrt(start, goal, obstacles, num_iterations, step_size, sampling_method)

# Find the correct path
correct_path = []
node = path[-1] if goal_reached else None
while node is not None:
    correct_path.append(node.point)
    node = node.parent
correct_path = correct_path[::-1]  # Reverse to start from the beginning

# Find the smooth path
smoothed_path = smooth_path(correct_path, obstacles)

# Plotting the result
fig, ax = plt.subplots()
ax.set_xlim(0, space_size[0])
ax.set_ylim(0, space_size[1])
# Plot the obstacles
for obstacle in obstacles:
    x, y = obstacle.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='orange', ec='none')
# Plot the tree
for point in path:
    if point.parent is not None:
        plt.plot([point.point.x, point.parent.point.x], [point.point.y, point.parent.point.y], 'blue')
# Plot the points
for point in path:
    ax.plot(point.point.x, point.point.y, 'bo')
ax.plot(start[0], start[1], 'go', label='Start')
ax.plot(goal[0], goal[1], 'ro', label='Goal')
# Plot the correct path
for i in range(len(correct_path)-1):
    plt.plot([correct_path[i].x, correct_path[i+1].x], [correct_path[i].y, correct_path[i+1].y], 'm-', linewidth=3)
# Plot the smooth path
for i in range(len(smoothed_path)-1):
    plt.plot([smoothed_path[i].x, smoothed_path[i+1].x], [smoothed_path[i].y, smoothed_path[i+1].y], 'g-', linewidth=3)
# Show the plot
plt.legend(loc='upper left')
plt.title('RRT Path Planning')
plt.show()

"""
Informed RRT Sampling*: In Informed RRT* (an extension of RRT), once a path is found, further samples are drawn from an ellipsoidal region that contains all possible shorter paths. This focuses the search on areas that are most likely to improve the current solution.

Heuristic-Based Sampling: This approach uses heuristics or additional information about the environment (like gradients or potential fields) to guide the sampling process towards more promising areas.
"""

"""
Define the Path Problem:

Clearly define the initial and target points of your path.
Establish any constraints (e.g., obstacles, boundaries) and criteria for optimization (shortest distance, minimal turns, etc.).
Initial Path Generation:

Create an initial path. This can be a straight line between the start and end points or a more complex path based on available data.
Iterative Optimization:

Path Refinement: Adjust the path by moving intermediate points, ensuring it adheres to constraints and improves based on your criteria.
Cost Function: Implement a cost function to evaluate the efficiency of the path. This function should consider factors like distance, smoothness, and adherence to constraints.
Iterative Improvements: Use iterative methods to make incremental improvements to the path. This might involve moving path points slightly and checking if the cost function value improves.
Backtracking for Optimization:

Identify Problematic Sections: Use the cost function to identify parts of the path that are less optimal.
Backtrack and Re-route: For these sections, backtrack to a previous decision point and try alternative routes or adjustments.
Evaluate Alternatives: Use the cost function to evaluate these new alternatives. Keep the changes if they result in an improved path.
Smoothing the Path:

Curve Fitting: Apply curve fitting techniques (like Bezier curves or splines) to smooth out sharp turns or angles in the path.
Balance Between Smoothness and Accuracy: Ensure that while smoothing the path, you don’t deviate significantly from the optimal path.
Final Evaluation:

Once a satisfactory path is generated, perform a final evaluation using your cost function and ensure it meets all criteria and constraints.
Iterative Refinement (Optional):

If necessary, iterate over the entire process to further refine the path, especially if new data or constraints emerge.
"""