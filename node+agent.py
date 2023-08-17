import random


# Base class for any combatant (survivor or zombie)
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


# Survivor class derived from Agent, adds morale and healing capability
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
        
        # If zombies outnumber survivors by more than 2:1, try to move
        if len(current_node.zombies) > 2 * len(current_node.survivors):
            # Find safest neighboring node
            safest_node = min(current_node.connections, key=lambda node: len(node.zombies)/(len(node.survivors) + 1))  # +1 to prevent division by zero
            return safest_node
        
        # If current node has no resources, move to a node with resources
        if current_node.resources == 0:
            resource_nodes = [node for node in current_node.connections if node.resources > 0]
            if resource_nodes:
                return random.choice(resource_nodes)
        
        return None

# Specialized survivor class with increased combat skills
class WarriorSurvivor(Survivor):
    def __init__(self, name):
        super().__init__(name, health=100, attack_power=10, morale=100)
        self.combat_bonus = 2

    def attack(self):
        return super().attack() + self.combat_bonus


# Specialized survivor class with better resource collection
class ScavengerSurvivor(Survivor):
    def __init__(self, name):
        super().__init__(name, health=100, attack_power=10, morale=100)
        self.scavenging_bonus = 2

    def collect_resources(self, amount):
        super().collect_resources(amount + self.scavenging_bonus)


# Zombie class derived from Agent
class Zombie(Agent):

    def decide_movement(self, current_node):
        if not self.is_alive:
            return None

        # If there are no survivors in the current node, move towards nodes with most survivors
        if not current_node.survivors:
            most_survivors_node = max(current_node.connections, key=lambda node: len(node.survivors))
            if len(most_survivors_node.survivors) > 0:
                return most_survivors_node
            return random.choice(current_node.connections)
        
        return None


# Represents a location in the simulation
class Node:
    def __init__(self, name, terrain):
        self.name = name
        self.terrain = terrain
        self.connections = []
        self.survivors = []
        self.zombies = []
        self.resources = random.randint(0, 10)
        self.specialization = None

    def add_connection(self, node):
        if node not in self.connections:
            self.connections.append(node)
            node.connections.append(self)

    def add_survivor(self, survivor):
        self.survivors.append(survivor)

    def add_zombie(self, zombie):
        self.zombies.append(zombie)
        
    def specialize(self, specialization):
        self.specialization = specialization

    def terrain_impact(self, agent):
        terrain_effects = {
            ("Mountain", Survivor): 1.5,
            ("Forest", Zombie): 0.75,
            ("Forest", Survivor): 0.9,  # Example: slightly worse for survivors in a forest
        }
        return terrain_effects.get((self.terrain, type(agent)), 1)

    def random_events(self):
        pass

    def handle_event(self, event):
        pass

    def interact(self):
        self.random_events()

        for survivor in self.survivors[:]:
            move_to_node = survivor.decide_movement(self)
            if move_to_node:
                self.transfer_survivor(survivor, move_to_node)
            else:
                resources_collected = min(self.resources, 10)
                survivor.collect_resources(resources_collected)
                self.resources -= resources_collected
                self.resolve_combat(survivor)

        for zombie in self.zombies[:]:
            move_to_node = zombie.decide_movement(self)
            if move_to_node:
                self.transfer_zombie(zombie, move_to_node)

        self.apply_specialization_effects()

    def resolve_combat(self, survivor):
        for zombie in self.zombies[:]:
            if not survivor.is_alive:
                break
            
            # Survivor attacks zombie
            damage_to_zombie = survivor.attack() * self.terrain_impact(survivor)
            zombie.take_damage(damage_to_zombie)
            
            if not zombie.is_alive:
                self.zombies.remove(zombie)
                survivor.morale = min(100, survivor.morale + 10)  # Boost morale by 10
                continue
            
            # Zombie attacks survivor if it is still alive
            damage_to_survivor = zombie.attack() * self.terrain_impact(zombie)
            survivor.take_damage(damage_to_survivor)
            
            if not survivor.is_alive:
                self.survivors.remove(survivor)
                for other_survivor in self.survivors:  # Morale drop for other survivors
                    other_survivor.morale = max(0, other_survivor.morale - 10)

    def apply_specialization_effects(self):
        if self.specialization == 'hospital':
            for survivor in self.survivors:
                survivor.heal(5)
        elif self.specialization == 'armory':
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

    def __str__(self):
        return self.name


# Simulation class to handle setting up and running the game
class Simulation:
    def __init__(self, nodes, num_survivors, num_zombies):
        self.nodes = nodes
        self.setup(num_survivors, num_zombies)

    def setup(self, num_survivors, num_zombies):
        for _ in range(num_survivors):
            random.choice(self.nodes).add_survivor(Survivor(f"Survivor{_+1}", 100, 10, 100))

        for _ in range(num_zombies):
            random.choice(self.nodes).add_zombie(Zombie(f"Zombie{_+1}", 50, 8))

    def print_state(self):
        for node in self.nodes:
            print(f"{node.name} ({node.terrain}):")
            print(f"  Survivors: [{' '.join([s.name + '(M:' + str(s.morale) + ')' for s in node.survivors])}]")
            print(f"  Zombies: [{' '.join([z.name for z in node.zombies])}]")
            print(f"  Resources: {node.resources}")

    def run(self, num_iterations):
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}:")
            for node in self.nodes:
                node.interact()
            self.print_state()



import pygame

# Initialize pygame
pygame.init()

# Window settings
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Zombie Apocalypse Simulation")

# Colors and Fonts
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)  # Survivor
RED = (255, 0, 0)    # Zombie
BLUE = (0, 0, 255)   # Node
FONT = pygame.font.SysFont(None, 25)

def draw_node(node, x, y):
    # Scale node size based on resources
    size = 20 + 2 * node.resources
    pygame.draw.circle(win, BLUE, (x, y), size)
    text = FONT.render(node.name, True, WHITE)
    win.blit(text, (x - text.get_width()//2, y - text.get_height()//2))

def draw_agent(agent_color, x, y):
    pygame.draw.circle(win, agent_color, (x, y), 10)

def draw_connection(pos1, pos2):
    pygame.draw.line(win, WHITE, pos1, pos2, 3)

def print_state(nodes):
    print("="*60)
    for node in nodes:
        print(f"{node.name} ({node.terrain}):")
        print(f"  Survivors: [{' '.join([s.name + '(H:' + str(s.health) + ', M:' + str(s.morale) + ')' for s in node.survivors])}]")
        print(f"  Zombies: [{' '.join([z.name + '(H:' + str(z.health) + ')' for z in node.zombies])}]")
        print(f"  Resources: {node.resources}")
        print('-'*50)


# Example usage with integrated pygame rendering
if __name__ == "__main__":
    city_a = Node("City A", "Plains")
    city_b = Node("City B", "Forest")
    city_c = Node("City C", "Mountain")

    city_a.add_connection(city_b)
    city_b.add_connection(city_c)

    city_b.specialize('hospital')
    city_c.specialize('armory')

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
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Drawing
        win.fill((0, 0, 0))

        # Draw nodes and connections
        for node, position in node_positions.items():
            draw_node(node, *position)
            for connection in node.connections:
                draw_connection(position, node_positions[connection])

        # Draw survivors and zombies
        for node, position in node_positions.items():
            for i, survivor in enumerate(node.survivors):
                draw_agent(GREEN, position[0] + i*25, position[1] + 50)
            for i, zombie in enumerate(node.zombies):
                draw_agent(RED, position[0] + i*25, position[1] - 50)

        pygame.display.update()

        # Simulation Update and Printing
        sim.run(num_iterations=1)
        print_state(sim.nodes)

        pygame.time.delay(1000)


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
