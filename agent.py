import math
import random
import numpy as np
from dataclasses import dataclass, field, astuple
from typing import Any, Callable, Union

class Agent:
    """Represents an agent with an id, a position and health.
    
    Attributes:
        id (int): The id of the agent.
        position (tuple): The position of the agent on the map.
        health (int): The health of the agent.
    """
    def __init__(self, id, health, position):
        self.id = id
        self.health = health
        self.position = position # x, y coordinates on the map
        
        # health, strength, defense, speed attributes
        # speed controls who attacks first, if dead can't attack back
        # or not in turn-based game, attack in interval of speed time   
    
    def move(self, dx, dy):
        """Move the agent by a certain amount in the x and y directions.
        
        Args:
            dx (int): The amount to move in the x direction.
            dy (int): The amount to move in the y direction.
        """
        self.position = (self.position[0]+dx, self.position[1]+dy)


    def take_damage(self, damage):
        """Reduce the health of the agent by a certain amount.
        
        Args:
            damage (int): The amount of damage to take.
        """
        self.health -= damage
        
        # defend method to reduce damage taken
        
    def distance_to_agent(self, other_agent):
        """Calculate the distance between this agent and another agent.
    
        Args:
            other_agent (Agent): The other agent to calculate the distance to.
    
        Returns:
            float: The distance between the two agents.
        """
        x1, y1 = self.position
        x2, y2 = other_agent.position
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return int(distance)

class AgentManager:
    def __init__(self):
        self.agents = []
    
    def add_agent(self, agent):
        self.agents.append(agent)
    
    def remove_agent(self, agent):
        self.agents.remove(agent)
    
    def get_agents_in_range(self, agent, range):
        """Get a list of distances and agents within a certain range of a given agent.
    
        Args:
            agent (Agent): The agent to check the range from.
            range (float): The range to check for agents within.
        
        Returns:
            list: A list of distances and agents within the specified range.
        """
        agents_in_range = []
        for other_agent in self.agents:
            distance = agent.distance_to_agent(other_agent)
            if distance <= range:
                agents_in_range.append((distance, other_agent))
        agents_in_range.sort(key=lambda x: x[0])
        return agents_in_range

    def connect_agents(self, agent, all_agents):
        """Connect an agent to all other agents.
    
        Args:
            agent (Agent): The agent to connect.
            all_agents (list): A list of all agents.
        """
        agent.connections.append(all_agents)
        
    def add_connections_in_range(self, agent, range):
        """Add connections to all agents within a certain range of a given agent.
    
        Args:
            agent (Agent): The agent to add connections to.
            range (float): The range to check for agents within.
        """
        if len(agent.connections) == 0:
            agents_in_range = agent.get_agents_in_range(agent, range)
            for other_agent in agents_in_range:
                agent.connections.append(other_agent)
                other_agent.connections.append(agent)
        elif len(agent.connections) > 0:
            for other_agent in agent.connections:
                if agent.distance_to_agent(other_agent) > range:
                    agent.connections.remove(other_agent)
                    other_agent.connections.remove(agent)
            agents_in_range = agent.get_agents_in_range(agent, range)
            for other_agent in agents_in_range:
                if other_agent not in agent.connections:
                    agent.connections.append(other_agent)
                    other_agent.connections.append(agent)
                    
        # zombie may continue following a human even if other humans are closer




class Human(Agent):
    """Represents a human agent.
    
    Attributes:
        id (int): The id of the human.
        position (tuple): The position of the human on the map.
        health (int): The health of the human.
        weapon (Weapon): The weapon the human is currently holding.
    """
    def __init__(self, id, health, position, weapon=None):
        super().__init__(id, health, position)
        self._health = health
        self.weapon = weapon
        self.connections = []
        
    @property
    def health(self):
        return self._health
    
    @health.setter
    def health(self, value):
        self._health = value
        if self._health <= 0:
            print(f"Human {self.id} has died!")
            apocalypse.human_manager.remove_human(self)
            zombie = Zombie(len(apocalypse.zombie_manager.agents), 100, self.position)
            apocalypse.zombie_manager.add_zombie(zombie)
            
    """
    @cached_property
    def closest_enemy(self):
        return min(apocalypse.zombie_manager.zombies, key=lambda x: distance(self.position, x.position))
    """
        
    def take_turn(self, possible_weapons, closest_enemy):
        if closest_enemy is not None:
            self.attack(closest_enemy)
            # No zombies in range, so scavenge for supplies or move to a new position
        elif self.health < 50 or self.weapon is None or \
                any(weapon.damage > self.weapon.damage for weapon in possible_weapons):
                self.scavenge(possible_weapons)
        else:
            self.random_move()
            
    # move, then interact, then next turn

    def scavenge(self, possible_weapons):
        # Roll a dice to determine if the human finds any supplies
        if random.random() < 0.5:
            # Human has found some supplies
            supplies = random.randint(5, 10)
            self.health += supplies
        # Roll a dice to determine if the human finds a new weapon
        if random.random() < 0.2:
            # Human has found a new weapon
            new_weapon = random.choice(possible_weapons)
            if self.weapon is None or (new_weapon.damage * new_weapon.range > self.weapon.damage * self.weapon.range):
                self.weapon = new_weapon 
    
    def random_move(self):
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        self.move(dx, dy)
        
    def attack(self, zombie):
        # Calculate the damage dealt to the zombie
        damage = 10 if self.weapon == None else 10+self.weapon.damage
        # Deal the damage to the zombie
        zombie.take_damage(damage)

    """
    def attack(self, target):
        # if the target is a zombie, fight
            # if the human is faster than the zombie, they hit first
                # the zombie's health is reduced by the human's strength
                # if the zombie is still alive, attack back
                    # the human's health is reduced by the zombie's strength
                    # if the human is dead, turn to zombie
                # if the zombie is dead, die
            # if the zombie is faster than the human, they hit first
                # the human's health is reduced by the zombie's strength
                # if the human is still alive, attack back
                    # the zombie's health is reduced by the human's strength
                    # if the zombie is dead, die
                # if the human is dead, turn to zombie
    """
    """
    def attack_neighbors(self, agent):
        neighbors = self.get_neighbors(agent)
        if isinstance(agent, Human):
            for neighbor in neighbors:
                if isinstance(neighbor, Zombie):
                    self.attack_agent(agent, neighbor)
        elif isinstance(agent, Zombie):
            for neighbor in neighbors:
                if isinstance(neighbor, Human):
                    self.attack_agent(agent, neighbor)
    """
    
    # further processing in manager
    def take_damage(self, damage):
        # Reduce the human's health by the specified amount of damage
        super().take_damage(damage)
        


class HumanManager(AgentManager):
    """Manages the humans in the apocalypse.

    Attributes:
        humans (list): A list of all the humans in the apocalypse.
    """
    def __init__(self):
        self.agents = []
        
    def add_human(self, human):
        super().add_agent(human)
        
    def remove_human(self, human):
        super().remove_agent(human)

    def get_enemies_in_attack_range(self, agent):
        attack_range = math.sqrt(2+ agent.weapon.range) if (isinstance(agent, Human) and agent.weapon is not None) else math.sqrt(2)
        agents_in_range = super().get_agents_in_range(agent, attack_range)
        enemies_in_range = [(distance, enemy) for distance, enemy in agents_in_range if isinstance(enemy, Zombie)]
        return enemies_in_range
    
    def get_closest_enemy(self, agent):
        # Get the distance to each enemy
        enemies = self.get_enemies_in_attack_range(agent)
        if len(enemies) == 0:
            return None
        # Get the closest enemy
        closest_enemy = enemies[0]
        return closest_enemy
        
    def trade_with(self, agent1, agent2):
        if agent1.weapon is not None and agent2.weapon is not None:
            # agent with worse weapon give some health to agent with better weapon to trade
            if agent1.weapon < agent2.weapon and agent1.health > agent2.weapon.damage:
                agent1.weapon, agent2.weapon = agent2.weapon, agent1.weapon
                agent1.health -= agent1.weapon.damage
                agent2.health += agent1.weapon.damage
            elif agent1.weapon > agent2.weapon and agent1.health < agent2.weapon.damage:
                agent1.weapon, agent2.weapon = agent2.weapon, agent1.weapon
                agent2.health -= agent2.weapon.damage
                agent1.health += agent2.weapon.damage
            else:
                # if the weapons are the same, trade health
                if agent1.health > agent2.health:
                    agent1.health -= agent2.health
                    agent2.health += agent2.health
                else:
                    agent2.health -= agent1.health
                    agent1.health += agent1.health
        
                

    def print_human_info(self):
        print("Humans:")
        for human in self.agents:
            print(f"Human {human.id}: health={human.health}, position={human.position}, weapon={human.weapon}")


class Zombie(Agent):
    """Represents a zombie agent.
    
    Attributes:
        id (int): The id of the zombie.
        position (tuple): The position of the zombie on the map.
        health (int): The health of the zombie.
    """
    def __init__(self, id, health, position):
        super().__init__(id, health, position)
        self._health = health
        self.connections = []

    @property
    def health(self):
        return self._health
    
    @health.setter
    def health(self, value):
        self._health = value
        # Check if the zombie has been killed
        if self.health <= 0:
            # Remove the zombie from the list of zombies
            apocalypse.zombie_manager.remove_zombie(self)
            
    """
    @cached_property
    def closest_enemy(self):
        return min(apocalypse.zombie_manager.zombies, key=lambda x: distance(self.position, x.position))
    """

    def take_turn(self, closest_human):
        # If there are any humans in range, attack the closest one
        if closest_human is not None:
            if random.random() < 0.5:
                self.attack(closest_human)
            else:
                self.move_towards_human(closest_human)
        else:
            self.random_move()

    def attack(self, human):
        # Deal 10 damage to the human
        human.take_damage(20)

    # further processing in manager
    def take_damage(self, damage):
        super().take_damage(damage)
    
    def move_towards_human(self, human):
        # Calculate the direction in which the human is located
        dx, dy = self.calculate_direction(human)
        # Move the zombie towards the human
        self.move(dx, dy)

    def calculate_direction(self, human):
        dx = human.position[0] - self.position[0]
        dy = human.position[1] - self.position[1]
        return dx, dy
    
    def random_move(self):
        # Move the zombie to a random position
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        self.move(dx, dy)
        
    # check legal move
    
class ZombieManager(AgentManager):
    """Manages the zombies in the apocalypse.

    Attributes:
        zombies (list): A list of all the zombies in the apocalypse.
    """
    def __init__(self):
        self.agents = []
        
    def add_zombie(self, zombie):
        super().add_agent(zombie)
        
    def remove_zombie(self, zombie):
        super().remove_agent(zombie)

    def get_enemies_in_attack_range(self, agent):
        attack_range = math.sqrt(2+ agent.weapon.range) if (isinstance(agent, Human) and agent.weapon is not None) else math.sqrt(2)
        agents_in_range = super().get_agents_in_range(agent, attack_range)
        enemies_in_range = [(distance, enemy) for distance, enemy in agents_in_range if isinstance(enemy, Human)]
        return enemies_in_range
    
    def get_closest_enemy(self, agent):
        # Get the distance to each enemy
        enemies = self.get_enemies_in_attack_range(agent)
        if len(enemies) == 0:
            return None
        # Get the closest enemy
        closest_enemy = enemies[0]
        return closest_enemy
    
    def print_zombie_info(self):
        print("Zombies:")
        for zombie in self.agents:
            print(f"Zombie {zombie.id}: health={zombie.health}, position={zombie.position}")
    
# dataclass for weapon
@dataclass(order=True, frozen=True) # slots=True
class Weapon:
    """Represents a weapon that can be used by a human.
    
    Attributes:
        name (str): The name of the weapon.
        damage (int): The damage dealt by the weapon.
        range (int): The range of the weapon.
        trading_value (int): The trading value of the weapon, calculated as damage * range.
    """
    trading_value: int = field(init=False, repr=False)
    name: str
    damage: int = 0
    range: int = 0
    
    def __post_init__(self):
        object.__setattr__(self, "trading_value", self.damage * self.range)
        
    def __iter__(self):
        yield from astuple(self)
    
    def __str__(self):
        return f"{self.name} ({self.damage} damage, {self.range} range)"

class AgentFactory:
    
    def __init__(self):
        self.character_creation_funcs: dict[str, Callable[..., Agent]] = {}

    def register(self, character_type: str, creator_fn: Callable[..., Agent]) -> None:
        """Register a new game character type."""
        self.character_creation_funcs[character_type] = creator_fn

    def unregister(self, character_type: str) -> None:
        """Unregister a game character type."""
        self.character_creation_funcs.pop(character_type, None)

    def create(self, arguments: dict[str, Any]) -> Agent:
        """Create a game character of a specific type."""
        args_copy = arguments.copy()
        character_type = args_copy.pop("type")
        try:
            creator_func = self.character_creation_funcs[character_type]
        except KeyError:
            raise ValueError(f"unknown character type {character_type!r}") from None
        return creator_func(**args_copy)

class ZombieApocalypse:
    """Represents a zombie apocalypse and manages the humans and zombies.
    
    Attributes:
        map (list): A 2D list representing the map, with None representing a empty cell and a human or zombie representing a cell occupied by human or zombie.
        human_manager (HumanManager): The human manager instance to manage the humans.
        zombie_manager (ZombieManager): The zombie manager instance to manage the zombies.
    """
    def __init__(self, num_zombies, num_humans, school_size=5):
        self.map = [[None for _ in range(school_size)] for _ in range(school_size)]
        self.human_manager = HumanManager()
        self.zombie_manager = ZombieManager()
        self.factory = AgentFactory()
        self.initialize(num_zombies, num_humans, school_size)
        self.possible_weapons = [
            Weapon("Baseball Bat", 20, 2),
            Weapon("Pistol", 30, 5),
            Weapon("Rifle", 40, 8),
            Weapon("Molotov Cocktail", 50, 3),
        ]
        
    def initialize(self, num_zombies, num_humans, school_size):
        """Initializes the humans and zombies in the map.
        
        Args:
            num_zombies (int): The number of zombies to initialize.
            num_humans (int): The number of humans to initialize.
        """
        # Initialize humans and zombies
        self.factory.register("human", Human)
        self.factory.register("zombie", Zombie)
        for i in range(num_humans):
            self.human_manager.add_human(self.create_agent(school_size, i, "human"))
        for i in range(num_zombies):
            self.zombie_manager.add_zombie(self.create_agent(school_size, i, "zombie"))
            
    # break down the initialize method using factory pattern to separate the creation and the use for humans and zombies
    # factory pattern for humans and zombies
    def create_agent(self, school_size, id, type) -> Agent:
        arguments = {
            "type": type,
            "id": id,
            "health": 100,
            "location": (random.randint(0, school_size-1),random.randint(0, school_size-1))
        }
        return self.factory.create(arguments)
        
    # ensure legal location
    # factory pattern for weapons
    # factory pattern for map
    
            
    def simulate(self, num_turns):
        """Simulates the zombie apocalypse for the given number of turns.
        
        Args:
            num_turns (int): The number of turns to simulate.
        """
        # while number of humans > 0 and number of zombies > 0
        for i in range(num_turns):
            print(f"Turn {i+1}")
            self.human_manager.print_human_info()
            self.zombie_manager.print_zombie_info()
            print()
            self.take_turn()
            if len(self.human_manager.agents) == 0 or len(self.zombie_manager.agents) == 0:
                break
    # an escape position or method for humans to win
            
    def take_turn(self):
        """Simulates a turn in the zombie apocalypse. Each human and zombie takes a turn in the order they were initialized.
        """
        # Zombies take their turn
        for zombie in self.zombie_manager.agents:
            closest_human = self.zombie_manager.get_closest_enemy(zombie)
            zombie.take_turn(closest_human)
        # Humans take their turn
        for human in self.human_manager.agents:
            closest_zombie = self.human_manager.get_closest_enemy(human)
            human.take_turn(self.possible_weapons, closest_zombie)
    
    def move_agent(self, agent, dx, dy):
        """Move an agent on the map by changing its position attribute and updating the map list.
        
        Args:
            agent (Agent): The agent to move.
            dx (int): The change in x position.
            dy (int): The change in y position.
            
        Returns:
            bool: True if the agent was successfully moved, False otherwise.
        """
        if 0 <= agent.position[0] + dx < len(self.map) and 0 <= agent.position[1] + dy < len(self.map[0]):
            if self.map[agent.position[0] + dx][agent.position[1] + dy] == 0:
                agent.position = (agent.position[0] + dx, agent.position[1] + dy)
                self.map[agent.position[0]][agent.position[1]] = agent
                self.map[agent.position[0] - dx][agent.position[1] - dy] = None
                return True
        return False
    
    def print_map(self):
        """Print the map in a readable format.
        """
        for row in self.map:
            for cell in row:
                if cell is None:
                    print(" ", end=" ")
                elif isinstance(cell, Human):
                    print("H", end=" ")
                elif isinstance(cell, Zombie):
                    print("Z", end=" ")
            print()



apocalypse = ZombieApocalypse(5, 5)
apocalypse.simulate(10)


"""
take damage not working
wrongly moving more than one space
"""

"""
This revised code separates the simulation and its agents into separate classes. The ZombieApocalypse class stores instances of the HumanManager and ZombieManager classes, and has methods to advance the simulation by a single time step, add or remove humans and zombies from the simulation, and print a representation of the simulation map. The Human and Zombie classes both inherit from the Agent class and have additional attributes and methods specific to their roles in the simulation. The HumanManager and ZombieManager classes have methods to add and remove humans and zombies, respectively.
You can use this code to create and manipulate a simulation of a zombie apocalypse with humans and zombies. You can initialize the simulation with a certain number of humans and zombies using the ZombieApocalypse class, and advance the simulation by a single time step using its step method. You can add or remove humans and zombies from the simulation during runtime using the add_human and remove_human methods of the HumanManager class, and the add_zombie and remove_zombie methods of the ZombieManager class. You can also visualize the state of the simulation or retrieve information about the positions and attributes of the humans and zombies using the print_map method of the ZombieApocalypse class or the get_agents_within_range method of the Human or Zombie class.
"""