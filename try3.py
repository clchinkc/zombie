import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Settings
num_agents = 10
num_zombies = 3
num_survival_sites = 3
max_distance = 10
movement_step_size = 1.0
infection_radius = 1.5
immunity_radius = 2.0
buffer_zone = 1.0
site_crowding_thresholds = np.random.randint(3, 8, num_survival_sites)

# Initialize random seed
np.random.seed(0)

# Create survival sites with coordinates and crowding thresholds
survival_sites = pd.DataFrame({
    'site_id': range(num_survival_sites),
    'x': np.random.randint(0, max_distance, num_survival_sites),
    'y': np.random.randint(0, max_distance, num_survival_sites),
    'threshold': site_crowding_thresholds
})

# Create agents with initial positions and types
agents = pd.DataFrame({
    'agent_id': range(num_agents),
    'x': np.random.randint(0, max_distance, num_agents),
    'y': np.random.randint(0, max_distance, num_agents),
    'type': ['human' for _ in range(num_agents - num_zombies)] + ['zombie' for _ in range(num_zombies)]
})

# Function to calculate distances from agents to sites
def calculate_distances(agents, points):
    agent_coords = agents[['x', 'y']].to_numpy()
    point_coords = points[['x', 'y']].to_numpy()
    distances = np.sqrt(((agent_coords[:, np.newaxis, :] - point_coords)**2).sum(axis=2))
    return pd.DataFrame(distances, index=agents['agent_id'], columns=points['site_id'])

# Function to check if a point is within the immunity radius of any survival site
def is_within_safe_zone(x, y, survival_sites, immunity_radius):
    for _, site in survival_sites.iterrows():
        if ((x - site['x'])**2 + (y - site['y'])**2) <= immunity_radius**2:
            return True
    return False

# Function to move a zombie, preventing it from entering safe zones
def move_zombie(agent, survival_sites, step_size, immunity_radius):
    new_x, new_y = agent['x'] + np.random.randn() * step_size, agent['y'] + np.random.randn() * step_size
    if is_within_safe_zone(new_x, new_y, survival_sites, immunity_radius):
        return agent['x'], agent['y']
    return new_x, new_y

# Function to move an agent towards a target
def move_towards(agent, target, step_size):
    direction = np.array([target['x'] - agent['x'], target['y'] - agent['y']])
    distance = np.linalg.norm(direction)
    if distance <= step_size:
        return target['x'], target['y']
    else:
        move = direction / distance * step_size
        return agent['x'] + move[0], agent['y'] + move[1]

# Simulation function with infection logic
def simulate_movements(agents, survival_sites, steps=1, immunity_radius=immunity_radius):
    for step in range(steps):
        # Initialize site visitors count for this step
        site_visitors = {site_id: 0 for site_id in survival_sites['site_id']}

        for agent_id, agent in agents.iterrows():
            if agent['type'] == 'human':
                distances = calculate_distances(pd.DataFrame([agent]), survival_sites)
                adjusted_likelihood = 1 / distances**2
                for site_id, visitors in site_visitors.items():
                    if visitors >= survival_sites.loc[site_id, 'threshold']:
                        adjusted_likelihood[site_id] = 0
                target_site_id = adjusted_likelihood.idxmax(axis=1).iloc[0]
                target_site = survival_sites.loc[target_site_id]
                new_x, new_y = move_towards(agent, target_site, movement_step_size)
                # Update site visitors count
                if float(np.linalg.norm([new_x - target_site['x'], new_y - target_site['y']])) < movement_step_size:
                    site_visitors[target_site_id] += 1
            else:  # Zombie movement
                new_x, new_y = move_zombie(agent, survival_sites, movement_step_size, immunity_radius)
            agents.loc[agent_id, ['x', 'y']] = [new_x, new_y]

            # Infection logic
            if agent['type'] == 'human':
                for _, zombie in agents[agents['type'] == 'zombie'].iterrows():
                    if float(np.linalg.norm([zombie['x'] - new_x, zombie['y'] - new_y])) < infection_radius:
                        agents.loc[agent_id, 'type'] = 'zombie'
                        break

    return agents, survival_sites

# Function to plot the simulation state
def plot_simulation(agents, survival_sites, step, immunity_radius):
    plt.figure(figsize=(10, 8))
    human_agents = agents[agents['type'] == 'human']
    zombie_agents = agents[agents['type'] == 'zombie']
    plt.scatter(human_agents['x'], human_agents['y'], color='blue', label='Humans')
    plt.scatter(zombie_agents['x'], zombie_agents['y'], color='green', label='Zombies')
    plt.scatter(survival_sites['x'], survival_sites['y'], color='red', marker='*', s=100, label='Survival Sites')

    for _, site in survival_sites.iterrows():
        safe_zone = plt.Circle((site['x'], site['y']), immunity_radius, color='red', fill=False, linestyle='--')
        plt.gca().add_artist(safe_zone)
        humans_in_zone = human_agents[((human_agents['x'] - site['x'])**2 + (human_agents['y'] - site['y'])**2) <= immunity_radius**2]
        plt.annotate(f'Site {site["site_id"]} (Humans: {len(humans_in_zone)}/{site["threshold"]})', (site['x'], site['y']), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(f'Zombie Apocalypse Simulation Step {step}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the simulation
plot_simulation(agents, survival_sites, 0, immunity_radius)
for step in range(5):
    agents, survival_sites = simulate_movements(agents, survival_sites, steps=1, immunity_radius=immunity_radius)
    plot_simulation(agents, survival_sites, step + 1, immunity_radius)
