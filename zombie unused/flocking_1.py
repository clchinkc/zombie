
import numpy as np
import pygame


# Base Agent class
class Agent:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(2, dtype=float)
        self.best_position = np.copy(self.position)
        self.best_score = np.inf

    def move(self):
        self.position += self.velocity
        self.velocity += self.acceleration
        self.acceleration = np.zeros(2, dtype=float)  # Reset acceleration after each move

    def apply_force(self, force):
        self.acceleration += force

# FlockingAgent class inheriting from Agent
class FlockingAgent(Agent):
    MAX_SPEED = 5.0  # Defining a maximum speed for the agent
    
    def __init__(self, position, velocity):
        super().__init__(position, velocity)
        self.color = pygame.Color('blue')
        self.direction = np.random.uniform(-1, 1, 2)
        self.direction /= np.linalg.norm(self.direction)  # Normalize the direction vector

    def move(self):
        super().move()
        
        # Limit the velocity to the maximum speed
        speed = np.linalg.norm(self.velocity)
        if speed > FlockingAgent.MAX_SPEED:
            self.velocity = (self.velocity / speed) * FlockingAgent.MAX_SPEED
        
        if np.random.rand() < 0.1:  # Random direction change for unpredictability
            self.direction = np.random.uniform(-1, 1, 2)
            self.direction /= np.linalg.norm(self.direction)
            self.velocity += self.direction

# Define interaction rules for flocking behavior

def cohesion(agent, agents, cohesion_radius):
    """Agents move towards the center of mass of their neighbors."""
    center_of_mass = np.zeros(2)
    num_neighbors = 0
    
    for other_agent in agents:
        distance = np.linalg.norm(agent.position - other_agent.position)
        if other_agent != agent and distance < cohesion_radius:
            center_of_mass += other_agent.position
            num_neighbors += 1

    if num_neighbors:
        center_of_mass /= num_neighbors
        desired_velocity = center_of_mass - agent.position
        return desired_velocity
    else:
        return np.zeros(2)

def alignment(agent, agents, alignment_radius):
    """Agents align their velocities with the average velocity of their neighbors."""
    avg_velocity = np.zeros(2)
    num_neighbors = 0
    
    for other_agent in agents:
        distance = np.linalg.norm(agent.position - other_agent.position)
        if other_agent != agent and distance < alignment_radius:
            avg_velocity += other_agent.velocity
            num_neighbors += 1

    if num_neighbors:
        avg_velocity /= num_neighbors
        return avg_velocity
    else:
        return np.zeros(2)

def separation(agent, agents, separation_radius):
    """Agents maintain a minimum distance from each other to avoid collisions."""
    repulsion = np.zeros(2)
    num_neighbors = 0
    epsilon = 1e-10  # small value to prevent division by zero
    
    for other_agent in agents:
        distance = np.linalg.norm(agent.position - other_agent.position)
        if other_agent != agent and distance < separation_radius:
            diff = agent.position - other_agent.position
            repulsion += diff / (distance + epsilon)  # add epsilon to the denominator
            num_neighbors += 1

    if num_neighbors:
        repulsion /= num_neighbors
        return repulsion
    else:
        return np.zeros(2)


# Define a fitness function to evaluate flocking behavior
def flocking_fitness(agent, agents):
    c_radius, a_radius, s_radius = 5, 5, 2  # Radii for cohesion, alignment, and separation
    c_force = cohesion(agent, agents, c_radius)
    a_force = alignment(agent, agents, a_radius)
    s_force = separation(agent, agents, s_radius)

    # Combine forces with weights (can be adjusted for desired behavior)
    combined_force = 1.0 * c_force + 1.2 * a_force + 1.5 * s_force
    agent.apply_force(combined_force)

    # Fitness is inversely proportional to the magnitude of the combined force
    # (the closer the force is to zero, the better the agent is adhering to the flocking rules)
    # can highlight the agents not in sync with the flock with the worst fitness by changing the color
    return 1.0 / (1.0 + np.linalg.norm(combined_force))

# Game class for the simulation
class FlockingSimulation:
    def __init__(self, num_agents, max_iterations, grid_size):
        self.grid_size = grid_size
        self.agents = [FlockingAgent(self.random_position(), [0, 0]) for _ in range(num_agents)]
        self.max_iterations = max_iterations

    def random_position(self):
        if getattr(self, 'agents', None) is None:
            return np.random.uniform(0, self.grid_size, 2)
        position = np.random.uniform(0, self.grid_size, 2)
        while any(np.allclose(position, agent.position, atol=1.0) for agent in self.agents):
            position = np.random.uniform(0, self.grid_size, 2)
        return position

    def run(self):
        # Update the agents for a single iteration
        for agent in self.agents:
            fitness = flocking_fitness(agent, self.agents)
            agent.move()
            
            # Check for boundary collisions and bounce back if necessary
            for i in range(2):  # i=0 for x-axis, i=1 for y-axis
                agent.position[i] = max(0, min(agent.position[i], self.grid_size))
                if agent.position[i] == 0 or agent.position[i] == self.grid_size:
                    agent.velocity[i] *= -1

# Define a function to run the simulation and visualize the results
def visualize_simulation(num_agents, max_iterations, grid_size):
    pygame.init()
    screen = pygame.display.set_mode((grid_size, grid_size))
    clock = pygame.time.Clock()
    game = FlockingSimulation(num_agents, max_iterations, grid_size)

    running = True
    iteration = 0
    while running and iteration < max_iterations:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill(pygame.Color('black'))
        for agent in game.agents:
            pygame.draw.circle(screen, agent.color, agent.position.astype(int), 3)
        pygame.display.flip()
        clock.tick(30)
        game.run()
        iteration += 1

    pygame.quit()
    
# Run the simulation
visualize_simulation(num_agents=10, max_iterations=1000, grid_size=500)