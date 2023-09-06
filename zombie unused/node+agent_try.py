
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
        print(f"  Survivors: [{' '.join([s.name + '(H:' + str(s.health) + ', M:' + str(s.morale) + ')' for s in node.survivors])}]")
        print(f"  Zombies: [{' '.join([z.name + '(H:' + str(z.health) + ')' for z in node.zombies])}]")
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

def draw_agents(node_positions):
    for node, position in node_positions.items():
        num_agents = len(node.survivors) + len(node.zombies)
        for i, survivor in enumerate(node.survivors):
            draw_agent(GREEN, *position, offset=20, i=i, total=num_agents)
        for i, zombie in enumerate(node.zombies):
            draw_agent(RED, *position, offset=40, i=i, total=num_agents)

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
    
