import numpy as np


class Zombie:
    def __init__(self, position):
        self.position = position

    def move(self):
        # Zombies move randomly
        self.position = (self.position[0] + np.random.randint(-1, 2),
                         self.position[1] + np.random.randint(-1, 2))

class Survivor:
    def __init__(self, position, resources):
        self.position = position
        self.resources = resources

    def make_decision(self):
        # Survivors move randomly for now
        self.position = (self.position[0] + np.random.randint(-1, 2),
                         self.position[1] + np.random.randint(-1, 2))

class Environment:
    def __init__(self, weather, terrain):
        self.weather = weather
        self.terrain = terrain

    def update_environment(self):
        # Update environment here
        pass

class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def update(self, current_value):
        error = self.setpoint - current_value
        self.integral += error
        derivative = error - self.previous_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

# Initialize simulation parameters
grid_size = (10, 10)
initial_zombies = 50
initial_survivors = 20
initial_resources = 100
desired_survivor_ratio = 0.5  # Desired ratio of survivors to zombies
desired_resource_level = 150  # Desired level of resources

# Initialize zombies and survivors
zombies = [Zombie((np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))) for _ in range(initial_zombies)]
survivors = [Survivor((np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])), initial_resources) for _ in range(initial_survivors)]

# Initialize the environment
environment = Environment("Clear", "Urban")

# Initialize PID controllers for different objectives
pid_survivor_ratio = PIDController(1.0, 0.1, 0.05, desired_survivor_ratio)
pid_resource_management = PIDController(0.5, 0.05, 0.02, desired_resource_level)

# Function to update the simulation environment
def update_simulation(survivors, zombies, resources, environment):
    environment.update_environment()
    for survivor in survivors:
        survivor.make_decision()
    for zombie in zombies:
        zombie.move()

    current_ratio = len(survivors) / (len(zombies) + 1)  # Avoid division by zero
    ratio_control = pid_survivor_ratio.update(current_ratio)

    if environment.weather == "Stormy":
        ratio_control *= 0.5  # Less effective control during bad weather

    adjust_population(survivors, zombies, ratio_control)
    
    resource_control = pid_resource_management.update(resources)
    resources += resource_control - len(survivors) * 0.1

    return survivors, zombies, resources

# Function to adjust population based on PID control output
def adjust_population(survivors, zombies, ratio_control):
    # Logic to adjust population based on ratio_control
    survivor_change = int(ratio_control * 10)  # Example multiplier for effect
    zombie_change = -survivor_change

    while survivor_change > 0 and len(survivors) < grid_size[0] * grid_size[1]:
        survivors.append(Survivor((np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])), initial_resources))
        survivor_change -= 1
    while survivor_change < 0 and survivors:
        survivors.pop()
        survivor_change += 1

    while zombie_change > 0 and len(zombies) < grid_size[0] * grid_size[1]:
        zombies.append(Zombie((np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))))
        zombie_change -= 1
    while zombie_change < 0 and zombies:
        zombies.pop()
        zombie_change += 1

# Running the enhanced simulation
resources = initial_resources
for timestep in range(10):  # Run for 10 steps as an example
    survivors, zombies, resources = update_simulation(survivors, zombies, resources, environment)
    print(f"Time step {timestep}: Survivors: {len(survivors)}, Zombies: {len(zombies)}, Resources: {resources}")
