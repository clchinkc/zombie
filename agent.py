from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass, field
from typing import Any, Callable


class Agent(ABC):
    
    @abstractmethod
    def __init__(self, id: int, health: int, position: tuple[int, int]):
        self.id = id
        self.health = health
        self.position = position

    def move(self, dx, dy):
        self.position = (self.position[0]+dx, self.position[1]+dy)

    def take_damage(self, damage: int) -> None:
        self.health -= damage
    
    def distance_to_agent(self, other_agent: Agent) -> int:
        x1, y1 = self.position
        x2, y2 = other_agent.position
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return int(distance)
    
    @abstractmethod
    def take_turn(self, possible_weapons, closest_enemy):
        pass
    
    @abstractmethod
    def random_move(self):
        pass
    
    @abstractmethod
    def attack(self, enemy):
        pass
    
    @abstractmethod
    def scavenge(self, possible_weapons):
        pass

class AgentManager:
    def __init__(self):
        self.agents = []
    
    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)
    
    def remove_agent(self, agent: Agent) -> None:
        self.agents.remove(agent)
    
    def get_agents_in_range(self, agent: Agent, range: float):
        # Get a list of distances and agents within a certain range of a given agent.

        agents_in_range = []
        for other_agent in self.agents:
            distance = agent.distance_to_agent(other_agent)
            if distance <= range:
                agents_in_range.append((distance, other_agent))
        agents_in_range.sort(key=lambda x: x[0])
        return agents_in_range

    def connect_agents(self, agent: Any[Agent], all_agents: list[Agent]):
        # Connect an agent to the list of agents.
        agent.connections.extend(all_agents)
        
    def add_connections_in_range(self, agent: Any[Agent], range: float) -> None:
        # Add connections to all agents within a certain range of a given agent.
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
    def __init__(self, id, health, position, weapon=None):
        super().__init__(id, health, position)
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
    
    def move(self, dx, dy):
        super().move(dx, dy)

    def take_damage(self, damage: int) -> None:
        super().take_damage(damage)
        
    def distance_to_agent(self, other_agent: Agent) -> int:
        return super().distance_to_agent(other_agent)
        
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


# Manage all humans in the apocalypse
class HumanManager(AgentManager):
    
    def __init__(self):
        # List of all humans in the apocalypse
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
    
    def get_closest_zombie(self, agent):
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

    def __init__(self, id, health, position):
        super().__init__(id, health, position)
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
    
    def move(self, dx, dy):
        super().move(dx, dy)
        
    def take_damage(self, damage: int) -> None:
        super().take_damage(damage)
        
    def distance_to_agent(self, other_agent: Agent) -> int:
        return super().distance_to_agent(other_agent)

    def take_turn(self, closest_human):
        # If there are any humans in range, attack the closest one
        if closest_human is not None:
            if random.random() < 0.5:
                self.attack(closest_human)
            else:
                self.random_move()
    
    def random_move(self):
        # Move the zombie to a random position
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        self.move(dx, dy)
        
    # check legal move

    def attack(self, human):
        # Deal 10 damage to the human
        human.take_damage(20)
        
    def scavenge(self, possible_weapons):
        pass

# Manage all zombies in the apocalypse
class ZombieManager(AgentManager):

    def __init__(self):
        # List of all zombies in the apocalypse
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
    
    def get_closest_human(self, agent):
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
    
# dataclass for weapon that can be used by a human
@dataclass(order=True, frozen=True) # slots=True
class Weapon:

    trading_value: int = field(init=False, repr=False) # the value of the weapon in trading function
    name: str
    damage: int = 0
    range: int = 0
    
    def __post_init__(self):
        object.__setattr__(self, "trading_value", self.damage * self.range)
        
    def __iter__(self):
        yield from astuple(self)
    
    def __str__(self):
        return f"{self.name} ({self.damage} damage, {self.range} range)"

# Abstract Factory Pattern
class AbstractAgentFactory(ABC):
    @abstractmethod
    def create_agent(self, **kwargs) -> Agent:
        raise NotImplementedError()

class HumanFactory(AbstractAgentFactory):
    def create_agent(self, **kwargs) -> Agent:
        return Human(**kwargs)

class ZombieFactory(AbstractAgentFactory):
    def create_agent(self, **kwargs) -> Agent:
        return Zombie(**kwargs)

# Factory method Pattern
class AgentFactory:
    
    def __init__(self):
        self.character_creation_funcs: dict[str, Callable[..., AbstractAgentFactory]] = {}

    def register_character(self, character_type: str, creator_fn: Callable[..., AbstractAgentFactory]) -> None:
        """Register a new game character type."""
        self.character_creation_funcs[character_type] = creator_fn

    def unregister_character(self, character_type: str) -> None:
        """Unregister a game character type."""
        self.character_creation_funcs.pop(character_type, None)

    def produce(self, arguments: dict[str, Any]) -> Agent:
        """Create a game character of a specific type."""
        args_copy = arguments.copy()
        character_type = args_copy.pop("type")
        try:
            creator_func = self.character_creation_funcs[character_type]
        except KeyError:
            raise ValueError(f"unknown character type {character_type!r}") from None
        return creator_func.create_agent(**args_copy)

# may use builder pattern if the product is composite or if there are more parameters

# a zombie apocalypse and manages the humans and zombies
class ZombieApocalypse:

    def __init__(self, num_zombies, num_humans, school_size):
        self.map = [[None for _ in range(school_size)] for _ in range(school_size)] # a 2D list representing the map, with None representing a empty cell and a human or zombie representing a cell occupied by human or zombie
        self.human_manager = HumanManager() # the human manager instance to manage the humans
        self.zombie_manager = ZombieManager() # the zombie manager instance to manage the zombies
        self.factory = AgentFactory()
        self.initialize(num_zombies, num_humans, school_size)
        self.possible_weapons = [
            Weapon("Baseball Bat", 20, 2),
            Weapon("Pistol", 30, 5),
            Weapon("Rifle", 40, 8),
            Weapon("Molotov Cocktail", 50, 3),
        ]
        
    def initialize(self, num_zombies, num_humans, school_size):
        # Initializes the humans and zombies in the map.
        self.factory.register_character("human", HumanFactory)
        self.factory.register_character("zombie", ZombieFactory)
        for i in range(num_humans):
            self.human_manager.add_human(self.create_agent(school_size, i, "human"))
        for i in range(num_zombies):
            self.zombie_manager.add_zombie(self.create_agent(school_size, i, "zombie"))
            
    # separate the creation and the use for humans and zombies
    def create_agent(self, school_size, id, type) -> Agent:
        arguments = {
            "type": type,
            "id": id,
            "health": 100,
            "location": (random.randint(0, school_size-1),
                        random.randint(0, school_size-1))
        }
        return self.factory.produce(arguments)
        
    # ensure legal location
    # factory pattern for weapons
    # factory pattern for map
    
            
    def simulate(self, num_turns):
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
        # Simulates a turn in the zombie apocalypse
        # Zombies take their turn
        for zombie in self.zombie_manager.agents:
            closest_human = self.zombie_manager.get_closest_human(zombie)
            zombie.take_turn(closest_human)
        # Humans take their turn
        for human in self.human_manager.agents:
            closest_zombie = self.human_manager.get_closest_zombie(human)
            human.take_turn(self.possible_weapons, closest_zombie)
    
    def move_agent(self, agent, dx, dy):
        # Move an agent on the map by changing its position attribute and updating the map list.
        # return True if the agent was successfully moved, False otherwise.
        if 0 <= agent.position[0] + dx < len(self.map) and 0 <= agent.position[1] + dy < len(self.map[0]):
            if self.map[agent.position[0] + dx][agent.position[1] + dy] == 0:
                agent.position = (agent.position[0] + dx, agent.position[1] + dy)
                self.map[agent.position[0]][agent.position[1]] = agent
                self.map[agent.position[0] - dx][agent.position[1] - dy] = None
                return True
        return False
    
    def print_map(self):
        # Print the map in a readable format.
        for row in self.map:
            for cell in row:
                if cell is None:
                    print(" ", end=" ")
                elif isinstance(cell, Human):
                    print("H", end=" ")
                elif isinstance(cell, Zombie):
                    print("Z", end=" ")
            print()



apocalypse = ZombieApocalypse(5, 5, 5)
apocalypse.simulate(10)

"""
Change the inheritance to abstract class or protocol
"""

"""
take damage not working
wrongly moving more than one space
"""

"""
This revised code separates the simulation and its agents into separate classes. The ZombieApocalypse class stores instances of the HumanManager and ZombieManager classes, and has methods to advance the simulation by a single time step, add or remove humans and zombies from the simulation, and print a representation of the simulation map. The Human and Zombie classes both inherit from the Agent class and have additional attributes and methods specific to their roles in the simulation. The HumanManager and ZombieManager classes have methods to add and remove humans and zombies, respectively.
You can use this code to create and manipulate a simulation of a zombie apocalypse with humans and zombies. You can initialize the simulation with a certain number of humans and zombies using the ZombieApocalypse class, and advance the simulation by a single time step using its step method. You can add or remove humans and zombies from the simulation during runtime using the add_human and remove_human methods of the HumanManager class, and the add_zombie and remove_zombie methods of the ZombieManager class. You can also visualize the state of the simulation or retrieve information about the positions and attributes of the humans and zombies using the print_map method of the ZombieApocalypse class or the get_agents_within_range method of the Human or Zombie class.
"""