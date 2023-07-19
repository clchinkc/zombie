
import random


class Node:
    def __init__(self, name):
        self.name = name
        self.connections = []
        self.survivors = []
        self.zombies = []

    def add_connection(self, node):
        self.connections.append(node)

    def add_survivor(self, survivor):
        self.survivors.append(survivor)

    def add_zombie(self, zombie):
        self.zombies.append(zombie)

    def remove_survivor(self, survivor):
        self.survivors.remove(survivor)

    def remove_zombie(self, zombie):
        self.zombies.remove(zombie)

    def move_survivor(self, survivor, destination):
        if destination in self.connections:
            self.remove_survivor(survivor)
            destination.add_survivor(survivor)
            print(f"Survivor {survivor} moved from {self.name} to {destination.name}.")
        else:
            print(f"Cannot move survivor {survivor} to {destination.name}. Invalid destination.")

    def move_zombie(self, zombie, destination):
        if destination in self.connections:
            self.remove_zombie(zombie)
            destination.add_zombie(zombie)
            print(f"Zombie {zombie} moved from {self.name} to {destination.name}.")
        else:
            print(f"Cannot move zombie {zombie} to {destination.name}. Invalid destination.")

    def interact(self):
        for survivor in self.survivors:
            if survivor in self.zombies:
                self.remove_survivor(survivor)
                print(f"Survivor {survivor} was infected by a zombie and turned into a zombie!")
        for zombie in self.zombies:
            if zombie in self.survivors:
                self.remove_zombie(zombie)
                print(f"Zombie {zombie} was killed by a survivor!")

    def __str__(self):
        return self.name


def create_city_graph():
    city_a = Node("City A")
    city_b = Node("City B")
    city_c = Node("City C")

    city_a.add_connection(city_b)
    city_b.add_connection(city_a)
    city_b.add_connection(city_c)
    city_c.add_connection(city_b)

    return [city_a, city_b, city_c]


def simulate_zombie_apocalypse(cities, num_survivors, num_zombies, num_iterations):
    for i in range(num_survivors):
        random_city = random.choice(cities)
        random_city.add_survivor(f"Survivor {i+1}")

    for i in range(num_zombies):
        random_city = random.choice(cities)
        random_city.add_zombie(f"Zombie {i+1}")

    print("Initial state:")
    print_state(cities)

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}:")
        for city in cities:
            city.interact()
            for survivor in city.survivors:
                city.move_survivor(survivor, random.choice(city.connections))
            for zombie in city.zombies:
                city.move_zombie(zombie, random.choice(city.connections))
        print_state(cities)


def print_state(cities):
    for city in cities:
        print(f"{city.name}:")
        print(f"  Survivors: {city.survivors}")
        print(f"  Zombies: {city.zombies}")


# Example usage
cities = create_city_graph()
simulate_zombie_apocalypse(cities, num_survivors=5, num_zombies=3, num_iterations=5)


# https://networkit.github.io/dev-docs/notebooks.html
# https://towardsdatascience.com/visualizing-networks-in-python-d70f4cbeb259
# https://towardsdatascience.com/introducing-jaal-interacting-with-network-made-easy-124173bb4fa
# https://www.geeksforgeeks.org/networkx-python-software-package-study-complex-networks/
# https://wiki.python.org/moin/PythonGraphLibraries
# https://www.analyticsvidhya.com/blog/2022/04/all-about-popular-graph-network-tools-in-natural-language-processing/

"""
A Node Network Simulation to simulate Zombie Apocalypse dynamics could be an exciting way to model complex interactions in a hypothetical environment. The system would encompass various types of nodes, each representing an entity in the world, such as humans, zombies, resources, and buildings. The relationships and interactions between these entities would be represented by edges in the graph.

Here's an outline of how such a simulation might be designed:

1. **Node Types**: The network will have several types of nodes, each representing a different entity in the system.

    - **Human Nodes**: These nodes would represent individual humans. They would have properties such as health, speed, strength, and resources. They could also have a status property, which could be 'healthy', 'infected', or 'immune'.

    - **Zombie Nodes**: Represent individual zombies. They would have properties like health, speed, and strength. Zombies could also have a status, which could be 'active' or 'inactive' (e.g., destroyed).

    - **Resource Nodes**: Represent essential resources like food, water, medicine, and weapons. Humans can collect these resources to increase their survival chances.

    - **Building Nodes**: These nodes could represent safe houses or zombie nests. Safe houses increase humans' survival chances, while zombie nests spawn new zombie nodes.

2. **Edge Types**: The network will also have different types of edges, representing relationships and interactions between the nodes.

    - **Human-Human Edges**: These could represent social connections between humans. The strength of the edge could indicate the likelihood of humans helping each other.

    - **Human-Zombie Edges**: Represent the proximity of a human to a zombie. The closer a human is to a zombie, the higher the risk of infection.

    - **Human-Resource Edges**: Represent the proximity of a human to a resource. The closer a resource, the easier it is for a human to acquire it.

    - **Human-Building Edges**: Show how close a human is to a safe house or a zombie nest. This could influence the human's likelihood of finding shelter or encountering a horde of zombies.

3. **Simulation Dynamics**: Here's how you might simulate the system over time.

    - **Movement**: Each time step, nodes move around the map. Human nodes could use a search algorithm to find resource nodes or safe houses while avoiding zombie nodes. Zombie nodes could use a different algorithm to pursue human nodes.

    - **Infection**: If a human node gets too close to a zombie node, there's a chance the human node will become infected and transform into a new zombie node after some time.

    - **Resource Collection**: If a human node gets close enough to a resource node, they will collect the resource, which could improve their health, speed, strength, or provide immunity.

    - **Survival and Destruction**: Humans could use resources to defend against zombies. If a zombie's health falls to zero due to human attack, the zombie node becomes inactive.

4. **Simulation Metrics**: We could monitor various metrics to understand the simulation's progress.

    - **Human Count**: The number of human nodes still in the simulation.
    
    - **Zombie Count**: The number of active zombie nodes.
    
    - **Resource Distribution**: The number and distribution of resources in the simulation.
    
    - **Infection Rate**: The rate at which human nodes become infected.
    
    - **Survival Rate**: The rate at which human nodes survive from time step to time step.

With this setup, the simulation can be run for a certain number of time steps, or until all human nodes have been eliminated or have survived the zombie apocalypse. It can provide valuable insights into the behavior of complex systems and could even have real-world applications for understanding the spread of diseases or the dynamics of social networks.
"""
