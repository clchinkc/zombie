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

def steer(from_node, to_point, step_size, goal, obstacles):
    direction = np.array(to_point) - np.array((from_node.point.x, from_node.point.y))
    length = np.linalg.norm(direction)
    direction = direction / length if length != 0 else direction

    step = step_size
    new_point = np.array((from_node.point.x, from_node.point.y)) + step * direction
    return new_point if not is_collision(from_node.point, Point(new_point), obstacles) else None


def dynamic_step_size(from_node, to_point, step_size, goal, obstacles):
    from_coords = (from_node.point.x, from_node.point.y)  # Extract coordinates from the TreeNode

    goal_distance = np.linalg.norm(np.array(to_point) - np.array(goal))
    step = min(step_size, goal_distance)

    obstacle_distances = [obstacle.distance(Point(from_coords)) for obstacle in obstacles]
    if obstacle_distances:
        min_obstacle_distance = min(obstacle_distances)
        step = min(step, min_obstacle_distance / 2)  # Adjust as needed
    return step


def random_sampling(space_size):
    return np.random.uniform(0, space_size[0]), np.random.uniform(0, space_size[1])

def gaussian_sampling(space_size, goal, alpha=0.5):
    if np.random.rand() > alpha:
        return random_sampling(space_size)
    else:
        std_dev = min(space_size) / 6
        # Sample a point from a 2D Gaussian distribution
        while True:
            point = np.random.normal(goal, std_dev)
            if 0 <= point[0] <= space_size[0] and 0 <= point[1] <= space_size[1]:
                return point

def bridge_sampling(space_size, obstacles, bridge_length=5):
    while True:
        midpoint = random_sampling(space_size)
        direction = np.random.uniform(-1, 1, size=2)
        direction /= np.linalg.norm(direction)
        point_a = midpoint + bridge_length * direction / 2
        point_b = midpoint - bridge_length * direction / 2
        if not is_collision(Point(point_a), Point(point_b), obstacles) and 0 <= point_a[0] <= space_size[0] and 0 <= point_a[1] <= space_size[1] and 0 <= point_b[0] <= space_size[0] and 0 <= point_b[1] <= space_size[1]:
            return midpoint

def obstacle_sampling(space_size, obstacles, buffer=20.0, min_distance=1.0, alpha=0.5):
    if np.random.rand() > alpha:
        return random_sampling(space_size)
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
            if 0 <= point.x <= space_size[0] and 0 <= point.y <= space_size[1] and not obstacle.contains(point):
                return x, y

def informed_sampling(c_best, start, goal, space_size):
    if c_best == float('inf'):
        return random_sampling(space_size)
    
    center = np.array(start) + (np.array(goal) - np.array(start)) / 2
    a = c_best / 2  # Semi-major axis
    b = np.sqrt(a**2 - np.linalg.norm(np.array(goal) - np.array(start))**2 / 4)  # Semi-minor axis
    
    theta = np.arctan2((goal[1] - start[1]), (goal[0] - start[0]))  # Angle to rotate the axes
    
    while True:
        # Sample a random point in a unit circle then stretch it to the ellipse size
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, 1))  # To ensure uniform distribution
        p = np.array([a * r * np.cos(angle), b * r * np.sin(angle)])
        
        # Rotate the point by theta and translate it to the center
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rotated_p = np.array([
            cos_theta * p[0] - sin_theta * p[1],
            sin_theta * p[0] + cos_theta * p[1]
        ]) + center
        
        # Check if the point is within the space bounds
        if 0 <= rotated_p[0] <= space_size[0] and 0 <= rotated_p[1] <= space_size[1]:
            return rotated_p



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

    for _ in range(num_iterations):
        # Goal biasing: occasionally sample the goal
        if np.random.rand() < 0.1:  # 10% chance to sample the goal
            random_point = goal
        else:
            # Choose sampling method based on user input
            if sampling_method == 'informed':
                random_point = informed_sampling(c_best, start, goal, space_size)
            elif sampling_method == 'gaussian':
                random_point = gaussian_sampling(space_size, goal)
            elif sampling_method == 'bridge':
                random_point = bridge_sampling(space_size, obstacles)
            elif sampling_method == 'obstacle':
                random_point = obstacle_sampling(space_size, obstacles)
            else:
                random_point = random_sampling(space_size)

        nearest = nearest_node(tree, random_point)
        dynamic_size = dynamic_step_size(nearest, random_point, step_size, goal, obstacles)
        new_point_coords = steer(nearest, random_point, dynamic_size, goal, obstacles)
        if new_point_coords is not None:
            new_node = TreeNode(Point(new_point_coords), nearest)
            tree.append(new_node)
            if Point(new_point_coords).distance(Point(goal)) <= dynamic_size:
                goal_reached = True
                print("Reached the goal!")
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
num_iterations = 1500
step_size = 1.

# Choose the sampling method: 'random', 'gaussian', 'bridge', 'obstacle', or 'informed'
sampling_method = 'gaussian'

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
plt.title('RRT Path Planning (Correct Path in Magenta)')
plt.show()
