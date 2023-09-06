import math
import random
from collections import deque
from enum import Enum

import pygame

# Initialize pygame
pygame.init()

# Window
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
FONT = pygame.font.SysFont("Arial", 20)

# Window settings
win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Zombie Apocalypse Simulation")

# --- COLORS ---
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)  # Survivor
RED = (255, 0, 0)    # Zombie
BLUE = (0, 0, 255)   # Node

# --- ENUMERATIONS ---

class NodeType(Enum):
    DEFAULT = 1
    HOSPITAL = 2  # Acts as a safe house.
    ARMORY = 3
    RESOURCE = 4
    ZOMBIE_NEST = 5

# --- AGENT BASE CLASS ---

class Agent:
    def __init__(self, name, health, attack_power):
        self.name = name
        self.health = health
        self.attack_power = attack_power
        self.is_alive = True

    def attack(self):
        return random.randint(1, self.attack_power)

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            self.is_alive = False
            self.health = 0

    def decide_movement(self, current_node):
        return None

# --- SURVIVOR CLASSES ---

class Survivor(Agent):
    MAX_HEALTH = 100

    def __init__(self, name, health, attack_power, morale):
        super().__init__(name, health, attack_power)
        self.morale = morale

    def heal(self, amount):
        self.health += amount
        self.health = min(self.health, self.MAX_HEALTH)

    def collect_resources(self, amount):
        self.morale += min(amount, 100 - self.morale)

    def decide_movement(self, current_node):
        if not self.is_alive:
            return None

        # If in a dangerous situation, decide based on morale
        if len(current_node.zombies) > 2 * len(current_node.survivors):
            if self.morale < 50: # Less confident survivors will try to flee
                safest_node = min(current_node.connections, key=lambda node: len(node.zombies) / (len(node.survivors) + 1))
                path = current_node.shortest_path_to(safest_node)
                if path and len(path) > 1:
                    return path[1]  # Return the next node on the path

        # If no resources, find a node with resources
        if current_node.resources == 0:
            resource_nodes = [node for node in current_node.reachable_nodes() if node.resources > 0]
            if resource_nodes:
                closest_resource_node = min(resource_nodes, key=lambda node: len(current_node.shortest_path_to(node)))
                path = current_node.shortest_path_to(closest_resource_node)
                if path and len(path) > 1:
                    return path[1]  # Return the next node on the path

        return None

    def __str__(self):
        return f"{self.name} (Health: {round(self.health, 1)}, Morale: {round(self.morale, 1)})"

class WarriorSurvivor(Survivor):
    def __init__(self, name):
        super().__init__(name, health=100, attack_power=10, morale=100)
        self.combat_bonus = 2

    def attack(self):
        return super().attack() + self.combat_bonus


class ScavengerSurvivor(Survivor):
    def __init__(self, name):
        super().__init__(name, health=100, attack_power=10, morale=100)
        self.scavenging_bonus = 2

    def collect_resources(self, amount):
        super().collect_resources(amount + self.scavenging_bonus)


# --- ZOMBIE CLASS ---

class Zombie(Agent):

    def decide_movement(self, current_node):
        if not self.is_alive:
            return None

        # Zombies are attracted to survivors
        if not current_node.survivors:
            target_node = max(current_node.connections, key=lambda node: len(node.survivors))
            if len(target_node.survivors) > 0:
                return target_node
        
        return None
    
    def __str__(self):
        return f"{self.name} (Health: {round(self.health, 1)})"

# --- RESOURCE CLASS ---

class Resource:
    def __init__(self, resource_type, quantity):
        self.resource_type = resource_type  # can be 'food', 'water', 'medicine', 'weapon'
        self.quantity = quantity

    def use(self, amount):
        self.quantity -= amount
        if self.quantity < 0:
            self.quantity = 0
        return amount

    def __str__(self):
        return f"{self.resource_type}: {self.quantity}"

# --- BUILDING CLASS ---

class Building:
    def __init__(self, building_type):
        self.building_type = building_type  # can be 'safe_house' or 'zombie_nest'
        if building_type == 'zombie_nest':
            self.spawn_rate = 1  # Number of zombies spawned per iteration

# --- NODE CLASS ---

class Node:
    def __init__(self, name, terrain, node_type=NodeType.DEFAULT, resources=None, building=None):
        self.name = name
        self.terrain = terrain
        self.connections = []
        self.survivors = []
        self.zombies = []
        self.node_type = node_type
        self.resources = resources
        self.building = building

    def add_connection(self, node):
        if node not in self.connections:
            self.connections.append(node)
            node.connections.append(self)

    def add_survivor(self, survivor):
        self.survivors.append(survivor)

    def add_zombie(self, zombie):
        self.zombies.append(zombie)

    def terrain_impact(self, agent):
        terrain_effects = {
            ("Mountain", Survivor): 1.5,
            ("Forest", Zombie): 0.75,
            ("Forest", Survivor): 0.9,
        }
        return terrain_effects.get((self.terrain, type(agent)), 1)

    def interact(self):
        for survivor in self.survivors[:]:
            move_to_node = survivor.decide_movement(self)
            if move_to_node:
                self.transfer_survivor(survivor, move_to_node)
            else:
                if self.resources and self.resources.quantity > 0:
                    resources_collected = min(self.resources.quantity, 10)
                    survivor.collect_resources(resources_collected)
                    self.resources.quantity = max(0, self.resources.quantity - resources_collected)
                self.resolve_combat(survivor)

        for zombie in self.zombies[:]:
            move_to_node = zombie.decide_movement(self)
            if move_to_node:
                self.transfer_zombie(zombie, move_to_node)

        self.apply_specialization_effects()
        
        if self.building and self.building.building_type == 'zombie_nest':
            for _ in range(self.building.spawn_rate):
                self.add_zombie(Zombie(f"Zombie{len(self.zombies) + 1}", 50, 8))

        if self.resources and self.resources.quantity > 0:
            for survivor in self.survivors:
                if self.resources.resource_type == 'medicine':
                    survivor.heal(self.resources.use(5))
                elif self.resources.resource_type == 'weapon':
                    survivor.attack_power = min(survivor.attack_power + self.resources.use(1), 20)
                elif self.resources.resource_type == 'food':
                    survivor.morale = min(100, survivor.morale + self.resources.use(5))

    def resolve_combat(self, survivor):
        for zombie in self.zombies[:]:
            if not survivor.is_alive:
                break

            damage_to_zombie = survivor.attack() * self.terrain_impact(survivor)
            print(f"Survivor {survivor.name} attacks Zombie {zombie.name} for {damage_to_zombie} damage.")  # Debug statement
            zombie.take_damage(damage_to_zombie)

            if not zombie.is_alive:
                print(f"Zombie {zombie.name} is defeated!")  # Debug statement
                self.zombies.remove(zombie)
                survivor.morale = min(100, survivor.morale + 10)
                continue

            damage_to_survivor = zombie.attack() * self.terrain_impact(zombie)
            print(f"Zombie {zombie.name} attacks Survivor {survivor.name} for {damage_to_survivor} damage.")  # Debug statement
            survivor.take_damage(damage_to_survivor)

            if not survivor.is_alive:
                print(f"Survivor {survivor.name} is defeated!")  # Debug statement
                self.survivors.remove(survivor)
                for other_survivor in self.survivors:
                    other_survivor.morale = max(0, other_survivor.morale - 10)


    def apply_specialization_effects(self):
        if self.node_type == NodeType.HOSPITAL:
            for survivor in self.survivors:
                survivor.heal(5)
        elif self.node_type == NodeType.ARMORY:
            for survivor in self.survivors:
                survivor.attack_power = min(survivor.attack_power + 5, 20)

    def transfer_survivor(self, survivor, to_node):
        if survivor in self.survivors:
            self.survivors.remove(survivor)
            to_node.add_survivor(survivor)

    def transfer_zombie(self, zombie, to_node):
        if zombie in self.zombies:
            self.zombies.remove(zombie)
            to_node.add_zombie(zombie)

    def reachable_nodes(self):
        """Get all nodes reachable from the current node."""
        visited = set()
        to_visit = [self]
        while to_visit:
            current = to_visit.pop()
            if current not in visited:
                visited.add(current)
                to_visit.extend([node for node in current.connections if node not in visited])
        return visited

    def shortest_path_to(self, target_node):
        """Find shortest path from this node to target_node using BFS."""
        visited = set()
        queue = deque([(self, [])])  # Each entry is current_node, path_so_far

        while queue:
            current_node, path = queue.popleft()

            if current_node == target_node:
                return path + [current_node]

            visited.add(current_node)

            for neighbor in current_node.connections:
                if neighbor not in visited:
                    queue.append((neighbor, path + [current_node]))

        return None  # If no path found, return None

    def __str__(self):
        return self.name


# --- SIMULATION CLASS ---

class Simulation:
    def __init__(self, nodes, num_survivors, num_zombies):
        self.nodes = nodes
        self.num_survivors_history = []
        self.setup(num_survivors, num_zombies)

    def setup(self, num_survivors, num_zombies):
        for _ in range(num_survivors):
            random.choice(self.nodes).add_survivor(Survivor(f"Survivor{_+1}", 100, 10, 100))
        for _ in range(num_zombies):
            random.choice(self.nodes).add_zombie(Zombie(f"Zombie{_+1}", 50, 8))

    def run(self, num_iterations):
        for _ in range(num_iterations):
            for node in self.nodes:
                node.interact()
            # Append to the list instead of assigning
            self.num_survivors_history.append(sum(len(node.survivors) for node in self.nodes))


# Drawing Functions
def draw_node(node, x, y):
    size = 20 + 2 * node.resources.quantity if node.resources else 20
    pygame.draw.circle(win, BLUE, (x, y), size)
    text = FONT.render(node.name, True, WHITE)
    win.blit(text, (x - text.get_width()//2, y - text.get_height()//2))

def draw_agent(agent_color, x, y, offset=0, i=0, total=1):
    angle = i * 2 * math.pi / total
    distance = 50 + offset
    dx = int(math.cos(angle) * distance)
    dy = int(math.sin(angle) * distance)
    pygame.draw.circle(win, agent_color, (x + dx, y + dy), 10)

def draw_connection(pos1, pos2):
    pygame.draw.line(win, WHITE, pos1, pos2, 3)

def draw_metrics(num_survivors):
    text = FONT.render(f"Survivors: {num_survivors}", True, WHITE)
    win.blit(text, (10, 10))

def print_state(nodes):
    print("="*60)
    for node in nodes:
        print(f"{node.name} ({node.terrain}):")
        print(f"  Survivors: [{' '.join(str(survivor) for survivor in node.survivors)}]")
        print(f"  Zombies: [{' '.join(str(zombie) for zombie in node.zombies)}]")
        print(f"  Resources: {node.resources}")
        print('-'*50)


# --- PYGAME MAIN LOOP FUNCTIONS ---

def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True

def clear_screen():
    win.fill((0, 0, 0))

def draw_nodes_and_connections(node_positions):
    for node, position in node_positions.items():
        draw_node(node, *position)
        for connection in node.connections:
            draw_connection(position, node_positions[connection])

def draw_agent_health(agent, x, y, offset=0, i=0, total=1):
    angle = i * 2 * math.pi / total
    distance = 50 + offset
    dx = int(math.cos(angle) * distance)
    dy = int(math.sin(angle) * distance)
    
    health_text = FONT.render(str(round(agent.health)), True, WHITE)
    win.blit(health_text, (x + dx - health_text.get_width()//2, y + dy - health_text.get_height() - 15))

def draw_agents(node_positions):
    for node, position in node_positions.items():
        num_agents = len(node.survivors) + len(node.zombies)
        for i, survivor in enumerate(node.survivors):
            draw_agent(GREEN, *position, offset=20, i=i, total=num_agents)
            draw_agent_health(survivor, *position, offset=20, i=i, total=num_agents)
        for i, zombie in enumerate(node.zombies):
            draw_agent(RED, *position, offset=40, i=i, total=num_agents)
            draw_agent_health(zombie, *position, offset=40, i=i, total=num_agents)


def draw_simulation_metrics(sim):
    if sim.num_survivors_history:
        draw_metrics(sim.num_survivors_history[-1])


# --- MAIN LOOP ---

if __name__ == "__main__":
    # Define resources
    food_resource = Resource('food', 100)
    medicine_resource = Resource('medicine', 50)

    # Define buildings
    zombie_nest = Building('zombie_nest')

    # Define and connect nodes with resources and buildings
    city_a = Node("City A", "Plains", resources=food_resource)
    city_b = Node("City B", "Forest", NodeType.HOSPITAL)
    city_c = Node("City C", "Mountain", NodeType.ARMORY, building=zombie_nest)

    city_a.add_connection(city_b)
    city_b.add_connection(city_c)

    cities = [city_a, city_b, city_c]
    sim = Simulation(cities, num_survivors=5, num_zombies=3)

    # Define node positions
    node_positions = {
        city_a: (100, 100),
        city_b: (300, 500),
        city_c: (500, 100)
    }

    running = True
    while running:
        running = handle_events()

        clear_screen()
        
        draw_nodes_and_connections(node_positions)
        draw_agents(node_positions)
        draw_simulation_metrics(sim)
        
        pygame.display.update()
        
        print_state(cities)

        sim.run(num_iterations=1)
        
        if not any(len(city.survivors) for city in cities) or not any(len(city.zombies) for city in cities):
            running = False
        
        pygame.time.delay(1000)

    pygame.quit()
    
    if any(len(city.survivors) for city in cities):
        print("Survivors win!")
    else:
        print("Zombies win!")
        
    print(f"Survivors remaining: {sum(len(city.survivors) for city in cities)}")
    print(f"Zombies remaining: {sum(len(city.zombies) for city in cities)}")
    print(f"Simulation history: {sim.num_survivors_history}")




"""
**Enhanced Zombie Apocalypse Simulation Using Agent-Based and Node Network Approaches**:

1. **Node Representation**: Use the Node and NodeNetwork classes to signify cities or locations and their interconnections. Each node will host varying numbers of survivors, zombies, and potential resources.

2. **Agent Design**: Design Survivor and Zombie classes with attributes such as health, attack power, and speed. Survivors can belong to different factions or groups.

3. **Movement Dynamics**: Every turn allows survivors to move between connected nodes. Zombies, on the other hand, spread to adjacent cities via these links.

4. **Combat Mechanics**: Whenever survivors and zombies share a node, combat ensues. Continuously monitor and adjust each agent's health and status based on the outcome.

5. **Terrain and Location Specialities**: Incorporate terrain effects like forests, rivers, and mountains that influence movement and battles. Design specialized nodes like safe houses, hospitals, and "necropolis" nodes, each providing unique bonuses, challenges, or spawning functions.

6. **Resource Management**: Scatter resources across nodes, permitting survivors to collect essentials that can boost health or morale.

7. **Morale System**: Keep track of the survivors' morale. Factors such as confronting zombies, presence of leaders, and resource accessibility can influence this. Low morale can lead to decreased combat effectiveness or increased desertions.

8. **Diversified Units**: Introduce distinct survivor types like archers, mages, or cavalry, each carrying unique abilities to diversify combat and strategy.

9. **Victory Conditions**: The aim for survivors is to eradicate all zombie nodes or "necropolis." For zombies, the objective is to infect all survivors.

10. **Metrics and Overview**: Regularly update metrics like the human survival rate, zombie spread rate, and city occupation, offering players insights into the prevailing dynamics.

11. **Strategic Gameplay**: Let players strategize for human factions while the zombies remain AI-controlled. The topology of the node graph can be tailored, such as introducing pathways between distant nodes, to suit varied game scenarios.

12. **Expansion Potential**: To augment the depth of the simulation, consider adding elements like technology progression, diplomacy avenues between human factions, hero units, and more.

---

Please write the code referencing code 1 and code 2.

Please refactor and rewrite the code to optimize the code readibility.

Please provide high level suggestions for this zombie apocalypse simulation that combines agent-based approach and node approach.
"""


"""
Your simulation combines both agent-based and node-based models, a great strategy for modeling dynamic interactions in different environments. Below are some high-level suggestions to improve and enhance the current design:

1. **Enhance Modularity**:
    - Separate different functionalities into distinct modules or files. For instance, agent-related classes (`Agent`, `Survivor`, `Zombie`) can be in one module, while the node and simulation classes can be in another.

2. **Expand the Node System**:
    - Consider using a weighted graph system, so that moving from one node to another has a cost. This could affect decisions like whether to stay and fight or to move to another location.
    - Implement a pathfinding algorithm like A* for agents, so they can decide the best path to take based on their objectives.

3. **Introduce More Agent Types**:
    - Consider adding more specialized survivor types, like Medics (who can heal others) or Engineers (who can fortify locations).
    - Introduce varying zombie types, such as tanky, fast, or sneaky zombies.

4. **Enhance Interactions**:
    - Allow agents to form groups or teams, which can coordinate actions like defense or scavenging.
    - Let agents make decisions: e.g., a survivor may decide to flee to a different node if their current node is overrun by zombies.
    - Implement trading or resource sharing between survivors.

5. **Environmental Dynamics**:
    - Different nodes might have different resource regeneration rates.
    - Seasons or weather can affect both agent stats and resource availability.
  
6. **Specialization and Node Features**:
    - Apart from `hospital` and `armory`, consider adding other specializations, like a farm (for food), barricade (for defense), or a workshop (for crafting).
    - Nodes might get destroyed or become uninhabitable, forcing agents to move.

7. **More Random Events**:
    - Introduce a broader variety of random events, such as:
      - Survivor camps joining your nodes.
      - Natural disasters.
      - Bandits raiding resources.
      - Helicopter rescues in a certain node.

8. **Extend Combat Mechanics**:
    - Implement a defense stat for survivors which can mitigate the damage from zombies.
    - Have a range of attack powers rather than one fixed power to add unpredictability.

9. **Add a GUI**:
    - Implementing a graphical user interface can provide a visual representation of nodes, their connections, agents, resources, and events. This can make the simulation more user-friendly and engaging.

10. **Feedback and Learning**:
    - Allow agents to learn from previous iterations, improving their decision-making over time using techniques like reinforcement learning.

11. **Extend Setup**:
    - Make the setup more dynamic. Instead of a fixed number of survivors and zombies, maybe have a range, or base it on the number of nodes. This way, as you expand the world, the setup can scale appropriately.

12. **Configuration and Parameters**:
    - Consider externalizing configuration parameters (like the number of agents, nodes, iterations, etc.) so that they can be tweaked without modifying the code.

Remember, the key is to iteratively build and test. Don't try to incorporate all suggestions at once, but rather prioritize based on your goals for the simulation.
"""


"""
Your zombie apocalypse simulation combines the agent-based model (ABM) and node-based model quite seamlessly. Here's a step-by-step breakdown of how we can enhance and fix it:

Here are a few enhancements and fixes we can apply:

1. **Make Simulation Turn-based**: The pygame loop runs continuously. To better see the simulation, it might be beneficial to make the simulation turn-based, where each turn proceeds either by a button press or a timer.

2. **Enhance Zombie Movement**: Currently, if zombies don't find any survivors in their node, they just randomly move. A more strategic move would be for zombies to move closer to the largest group of survivors. 

3. **Better Terrain Effects**: The `terrain_impact` is a great idea but needs to be expanded. For example, mountains might be difficult for both survivors and zombies, forests might provide cover for survivors, etc.

4. **Balancing**: Like all simulations, it'll need some tweaking to ensure neither survivors nor zombies are too overpowered. For instance, currently, when a survivor dies, other survivors lose morale, but they should also have a chance to flee.

5. **Dynamic Node Visualization**: Nodes could change color based on the balance of survivors to zombies or show some visual cue when resources are depleted.

6. **Inter-node Movement**: Allow survivors and zombies to move between nodes based on certain conditions. For example, a survivor might move to another node if resources are depleted, or there are too many zombies.

7. **Specialized Survivors**: The specialized survivors (Warrior and Scavenger) aren't currently used in the simulation. Integrate them to add more variety.

8. **Pygame Visual Enhancements**: Use images for zombies and survivors instead of circles. Additionally, provide UI elements to control the simulation (e.g., pause, speed up, restart).

9. **Interactive Elements**: Allow the user to click on a node to see detailed statistics or even drop additional resources or "reinforcements" into a node.

10. **Expand Random Events**: The function `random_events` exists but isn't implemented. Add events like "a horde of zombies appears" or "supply drop" that can change the course of the simulation.

11. **Add Sound Effects**: Use pygame's mixer to add sounds, for instance, when zombies attack or when a node is under siege.

After identifying the areas of enhancement, each can be approached step by step. Starting with pygame improvements could provide a better visualization tool, which then makes testing and tweaking the core logic easier.

Would you like help with a specific section or enhancement?
"""


"""
You can expand on this foundation by adding more features, mechanics, and elements, such as the ability for survivors to form groups, more complex terrains, the introduction of different types of zombies, building defenses, etc.
"""



"""
Node Network Simulation

# https://networkit.github.io/dev-docs/notebooks.html
# https://towardsdatascience.com/visualizing-networks-in-python-d70f4cbeb259
# https://towardsdatascience.com/introducing-jaal-interacting-with-network-made-easy-124173bb4fa
# https://www.geeksforgeeks.org/networkx-python-software-package-study-complex-networks/
# https://wiki.python.org/moin/PythonGraphLibraries
# https://www.analyticsvidhya.com/blog/2022/04/all-about-popular-graph-network-tools-in-natural-language-processing/


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