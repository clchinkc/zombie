
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Protocol, Union


class Position(NamedTuple):
    x: int
    y: int


class HumanProtocol(Protocol):
    def scavenge(self, possible_weapons):
        pass
    
    def use_item(self, item):
        pass


class Agent(ABC):
    
    @abstractmethod
    def __init__(self, id: int, position: Position, health: int, strength: int, armour: int, speed: int):
        self.id = id
        self.position = position
        self.health = health
        self.strength = strength
        self.armour = armour
        self.speed = speed

    def move(self, direction: tuple[int, int]):
        self.position = Position(self.position.x+direction[0], 
                                self.position.y+direction[1])
    
    def distance_to_agent(self, other_agent: Agent) -> int:
        x1, y1 = self.position
        x2, y2 = other_agent.position
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return int(distance)

    def take_damage(self, damage: Union[int, float]):
        self.health -= damage
    
    @abstractmethod
    def take_turn(self, possible_weapons, closest_enemy):
        pass
    
    @abstractmethod
    def random_move(self):
        pass
    
    @abstractmethod
    def attack(self, enemy):
        pass
    
    def __str__(self) -> str:
        info = f"{self.__class__.__name__}\n"
        for var in vars(self):
            info += f"{var}: {getattr(self, var)}\n"
        return info
    
    def __repr__(self) -> str:
        return "{classname}({variables})".format(classname=self.__class__.__name__, variables=", ".join([str(getattr(self, var)) for var in vars(self)]))

class AgentManager(ABC):
    
    @abstractmethod
    def __init__(self):
        self.zombies = []
    
    @abstractmethod
    def add_agent(self, agent: Agent) -> None:
        pass
    
    @abstractmethod
    def remove_agent(self, agent: Agent) -> None:
        pass

    def get_agents_in_range(self, agent: Agent, range: float):
        # Get a list of distances and agents within a certain range of a given agent.
        agents_in_range = []
        for other_agent in self.zombies:
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
        _, agents_in_range = self.get_agents_in_range(agent, range)
        self.connect_agents(agent, agents_in_range)

        # zombie may continue following a human even if other humans are closer


class Human(Agent):
    def __init__(self, id, position: Position, health: int, strength: int, armour: int, speed: int, intelligence: int, weapon: Union[Weapon, None]=None):
        super().__init__(id, position, health, strength, armour, speed)
        self.intelligence = intelligence
        self.inventory = []
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
            apocalypse.human_manager.remove_agent(self)
            zombie = Zombie(len(apocalypse.zombie_manager.zombies), Position(self.position.x, self.position.y), 100, 10, 0, 1)
            apocalypse.zombie_manager.add_agent(zombie)
            
    @property
    def defense(self):
        return self.armour*random.uniform(0.5, 1.5)
    
    def move(self, direction):
        return super().move(direction)
        
    def distance_to_agent(self, other_agent: Agent) -> int:
        return super().distance_to_agent(other_agent)

    def take_damage(self, damage: Union[int, float]):
        return super().take_damage(damage-self.defense)
        
    def take_turn(self, possible_weapons, closest_enemy):
        if closest_enemy is not None:
            self.attack(closest_enemy)
            # No zombies in range, so scavenge for supplies or move to a new position
        elif self.health < 50 or self.weapon is None or \
                any(weapon.damage > self.weapon.damage for weapon in possible_weapons):
                self.scavenge(possible_weapons)
        else:
            self.random_move()
            
    def random_move(self):
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        self.move((dx, dy))
        
    def attack(self, zombie):
        # Calculate the damage dealt to the zombie
        damage = self.strength*random.uniform(0.5, 1.5)
        damage += self.weapon.damage if self.weapon is not None else 0
        # Deal the damage to the zombie
        zombie.take_damage(damage)

    def scavenge(self, possible_weapons):
        # Roll a dice to determine if the human finds any supplies
        if random.random() < 0.5:
            # Human has found some supplies
            supplies = PowerUp(self.position, 10, 0, 0, 0, 0)
            self.inventory.append(supplies)
        # Roll a dice to determine if the human finds a new weapon
        if random.random() < 0.2:
            # Human has found a new weapon
            new_weapon = random.choice(possible_weapons)
            if self.weapon is None or (new_weapon.damage * new_weapon.range > self.weapon.damage * self.weapon.range):
                self.weapon = new_weapon

    def use_item(self, item):
        if isinstance(item, Weapon):
            self.weapon = item
            self.inventory.remove(item)
        elif isinstance(item, PowerUp):
            self.health += item.health
            self.strength += item.strength
            self.armour += item.armour
            self.speed += item.speed
            self.intelligence += item.intelligence
            self.inventory.remove(item)


# Manage all humans in the apocalypse
class HumanManager(AgentManager):
    
    def __init__(self):
        # List of all humans in the apocalypse
        self.humans = []
        
    def add_agent(self, human):
        self.humans.append(human)
        
    def remove_agent(self, human):
        self.humans.remove(human)
        
    def get_agents_in_range(self, agent: Agent, range: float):
        return super().get_agents_in_range(agent, range)

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
        for human in self.humans:
            print(f"Human {human.id}: health={human.health}, position={human.position}, weapon={human.weapon}")


class Zombie(Agent):

    def __init__(self, id, position: Position, health: int, strength: int, armour: int, speed: int):
        super().__init__(id, position, health, strength, armour, speed)
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
            apocalypse.zombie_manager.remove_agent(self)

    @property
    def defense(self):
        return self.armour*random.uniform(0.5, 1.5)
    
    def move(self, direction):
        return super().move(direction)
        
    def distance_to_agent(self, other_agent: Agent) -> int:
        return super().distance_to_agent(other_agent)
        
    def take_damage(self, damage: Union[int, float]):
        return super().take_damage(damage-self.defense)

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
        self.move((dx, dy))

    def attack(self, human):
        damage = self.strength*random.uniform(0.5, 1.5)
        # Deal 10 damage to the human
        human.take_damage(damage)

# Manage all zombies in the apocalypse
class ZombieManager(AgentManager):

    def __init__(self):
        # List of all zombies in the apocalypse
        self.zombies = []

    def add_agent(self, zombie):
        self.zombies.append(zombie)

    def remove_agent(self, zombie):
        self.zombies.remove(zombie)

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
        for zombie in self.zombies:
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
    
    def __str__(self):
        return f"Weapon {self.name}: {self.damage} damage, {self.range} range"

@dataclass(order=True, frozen=True) # slots=True
class PowerUp:
    
    position: tuple[int, int]
    health: int
    strength: int
    armour: int
    speed: int
    intelligence: int
    
    def __str__(self):
        return f"PowerUp at {self.position}: health={self.health}, strength={self.strength}, armour={self.armour}, speed={self.speed}, intelligence={self.intelligence}"

# Abstract Factory Pattern
class AbstractAgentFactory(ABC):
    @abstractmethod
    def create_agent(**kwargs) -> Agent:
        raise NotImplementedError()

# Builder Pattern
class HumanFactory(AbstractAgentFactory):
    def create_agent(**kwargs) -> Human:
        health = kwargs.get("health", 100)
        strength = kwargs.get("strength", 10)
        armour = kwargs.get("armour", 5)
        speed = kwargs.get("speed", 1)
        intelligence = kwargs.get("intelligence", 10)
        return Human(health=health, strength=strength, armour=armour, speed=speed, intelligence=intelligence, **kwargs)

class ZombieFactory(AbstractAgentFactory):
    def create_agent(**kwargs) -> Zombie:
        health = kwargs.get("health", 100)
        strength = kwargs.get("strength", 10)
        armour = kwargs.get("armour", 5)
        speed = kwargs.get("speed", 1)
        return Zombie(health=health, strength=strength, armour=armour, speed=speed, **kwargs)

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

# store counter in factory
# may use builder pattern if the product is composite or if there are more parameters

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
    
    def __str__(self):
        return "\n".join(["".join([str(cell) for cell in row]) for row in self.grid])
    
    def __getitem__(self, position):
        return self.grid[position[0]][position[1]]
    
    def __setitem__(self, position, value):
        self.grid[position[0]][position[1]] = value
        
    def get_human(self, position) -> Human:
        if isinstance(self.grid[position], Human):
            return self.grid[position]
        else:
            raise Exception('No human at this position')
        
    def get_zombie(self, position) -> Zombie:
        if isinstance(self.grid[position], Zombie):
            return self.grid[position]
        else:
            raise Exception('No zombie at this position')
        
    def get_item(self, position) -> Union[Weapon, PowerUp]:
        if isinstance(self.grid[position], Weapon) or isinstance(self.grid[position], PowerUp):
            return self.grid[position]
        else:
            raise Exception('No item at this position')

    def print_map(self):
        # Print the map in a readable format.
        for row in self.grid:
            for cell in row:
                if cell is None:
                    print(" ", end=" ")
                elif isinstance(cell, Human):
                    print("H", end=" ")
                elif isinstance(cell, Zombie):
                    print("Z", end=" ")
            print()

class Game(ABC):

        # template method pattern
        @abstractmethod
        def __init__(self, grid, agentfactory):
            self.grid = grid
            self.agentfactory = agentfactory
            self.agents = []
        
        @abstractmethod
        def simulate(self, agent_count, num_turns):
            self.initialize(agent_count)
            self.run(num_turns)
            self.grid.print_map()
        
        @abstractmethod
        def create_agent(self, **kwargs):
            return self.agentfactory.produce(**kwargs)
        
        @abstractmethod
        def initialize(self, agent_count):
            for _ in range(agent_count):
                self.agents.append(self.create_agent())
                self.grid.append(self.create_agent())
        
        @abstractmethod
        def run(self, num_turns):
            for _ in range(num_turns):
                for agent in self.agents:
                    agent.take_turn()
                if self.end_condition():
                    break
        
        @abstractmethod
        def end_condition(self, **kwargs):
            return len(self.agents) == 0

# a zombie apocalypse and manages the humans and zombies
class ZombieApocalypse(Game):

    def __init__(self):
        self.human_manager = HumanManager() # the human manager instance to manage the humans
        self.zombie_manager = ZombieManager() # the zombie manager instance to manage the zombies
        self.factory = AgentFactory()
        self.possible_weapons = [
            Weapon("Baseball Bat", 20, 2),
            Weapon("Pistol", 30, 5),
            Weapon("Rifle", 40, 8),
            Weapon("Molotov Cocktail", 50, 3),
        ]
    
    def add_human(self, human):
        self.grid[human.position] = human
        self.human_manager.add_agent(human)
        
    def remove_human(self, agent):
        self.grid[agent.position] = None
        self.human_manager.remove_agent(agent)
        
    def add_zombie(self, zombie):
        self.grid[zombie.position] = zombie
        self.zombie_manager.add_agent(zombie)
        
    def remove_zombie(self, zombie):
        self.grid[zombie.position] = None
        self.zombie_manager.remove_agent(zombie)
        
    def add_item(self, item):
        self.grid[item.position] = item
        
    def remove_item(self, item):
        self.grid[item.position] = None
    
    def simulate(self, num_zombies, num_humans, school_size, num_turns):
        # Simulates the game.
        self.initialize(num_zombies, num_humans, school_size)
        self.run(num_turns)
        self.human_manager.print_human_info()
        self.zombie_manager.print_zombie_info()
        self.grid.print_map()
    
    # separate the creation and the use for humans and zombies
    def create_agent(self, id, school_size, type) -> Agent:
        arguments = {
            "type": type,
            "id": id,
            "position": (random.randint(0, school_size-1),
                        random.randint(0, school_size-1))
        }
        return self.factory.produce(arguments)
        
    # ensure legal position
    # factory pattern for weapons
    # factory pattern for map
        
    def initialize(self, num_zombies, num_humans, school_size):
        # a 2D list representing the map, with None representing a empty cell and a human or zombie representing a cell occupied by human or zombie
        self.grid = Grid(school_size, school_size)
        # Initializes the humans and zombies in the map
        self.factory.register_character("human", HumanFactory)
        self.factory.register_character("zombie", ZombieFactory)
        for i in range(num_humans):
            self.human_manager.add_agent(self.create_agent(i, school_size, "human"))
        for i in range(num_zombies):
            self.zombie_manager.add_agent(self.create_agent(i, school_size, "zombie"))
    
    # initialise weapon and map
    
    def run(self, num_turns):
        # while number of humans > 0 and number of zombies > 0
        for i in range(num_turns):
            print(f"Turn {i+1}")
            # Simulates a turn in the zombie apocalypse
            # Zombies take their turn
            for zombie in self.zombie_manager.zombies:
                closest_human = self.zombie_manager.get_closest_human(zombie)
                zombie.take_turn(closest_human)
            # Humans take their turn
            for human in self.human_manager.humans:
                closest_zombie = self.human_manager.get_closest_zombie(human)
                human.take_turn(self.possible_weapons, closest_zombie)
            # End the turn by removing dead humans and zombies from the map.
            for human in self.human_manager.humans:
                if human.health <= 0:
                    self.grid[human.position.x][human.position.y] = None
                    self.human_manager.humans.remove(human)
            for zombie in self.zombie_manager.zombies:
                if zombie.health <= 0:
                    self.grid[zombie.position.x][zombie.position.y] = None
                    self.zombie_manager.zombies.remove(zombie)
            # End the game if there are no more humans or zombies.
            if self.end_condition():
                break

    # move agent to a new position only if the new position is empty
    
    def escape(self, agent):
        # define a escape position for the humans to win
        pass
    
    def end_condition(self):
        return len(self.human_manager.humans) == 0 or len(self.zombie_manager.zombies) == 0



apocalypse = ZombieApocalypse()
apocalypse.simulate(num_zombies=10, 
                    num_humans=10, 
                    school_size=10, 
                    num_turns=10)


"""
take damage not working
wrongly moving more than one space
can use "in" to check if a variable is in a list and return a boolean
use abstract class to define agent class
use protocol to define human specific methods and zombie specific methods
Dataclass for cell in grid
Dict for inventory management
"""

"""
This revised code separates the simulation and its agents into separate classes. The ZombieApocalypse class stores instances of the HumanManager and ZombieManager classes, and has methods to advance the simulation by a single time step, add or remove humans and zombies from the simulation, and print a representation of the simulation map. The Human and Zombie classes both inherit from the Agent class and have additional attributes and methods specific to their roles in the simulation. The HumanManager and ZombieManager classes have methods to add and remove humans and zombies, respectively.
You can use this code to create and manipulate a simulation of a zombie apocalypse with humans and zombies. You can initialize the simulation with a certain number of humans and zombies using the ZombieApocalypse class, and advance the simulation by a single time step using its step method. You can add or remove humans and zombies from the simulation during runtime using the add_human and remove_human methods of the HumanManager class, and the add_zombie and remove_zombie methods of the ZombieManager class. You can also visualize the state of the simulation or retrieve information about the positions and attributes of the humans and zombies using the print_map method of the ZombieApocalypse class or the get_agents_within_range method of the Human or Zombie class.
"""

"""
Immediate improvements:

agent class:
# 0's and 1's to represent the agent's genome
# the genome can be used to determine the agent's behavior
# state, health and size can be used to determine the agent's fitness
# position, direction, speed, energy, infection, infection time, death
# can be used to determine the agent's state

human class:
@cached_property
def closest_enemy(self):
    return min(apocalypse.zombie_manager.zombies, key=lambda x: distance(self.position, x.position))
def choose_action(self):
    # choose between move, attack, pick up item, use item
    # if low health, use item
    # elif there is enemy nearby, attack
    # elif there is item nearby, pick up
    # elif there is human nearby, trade
    # elif there is enough condition, craft item
    # else move
    pass
def take_turn(self):
    action = self.choose_action(human)
    If the action is "attack":
        Loop through the surrounding area of the human
    If there is a zombie in the surrounding area:
        The human attacks the zombie
        If the zombie's health is less than or equal to zero:
            Remove the zombie from the list of zombies
    If the action is "pick up":
        Loop through the surrounding area of the human
        If there is an item in the surrounding area:
            The human picks up the item
            Remove the item from the list of items
    If the action is "use":
        If the human's inventory is not empty:
            The human uses an item from the inventory
    If the action is "move":
        The human chooses a direction to move in
        Remove the human from the current position
        The human moves to the new position
        Add the human to the new position in the list of humans.
# choosing direction can go to a separate self.choose_direction(agent) method
# attack can go to a separate self.attack_neighbors(agent) method
# move, then interact, then next turn
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
scavenge(self, possible_weapons):
# PowerUp is created randomly and found
# not created when found

human manager class:
trade_with(self, agent1, agent2):
# may trade weapon and armour
# more clever conditions, based on intelligence
# trade different items, like specific amount of food for weapon
# craft method that takes food and turns it into item

zombie class:
@cached_property
def closest_enemy(self):
    return min(apocalypse.zombie_manager.zombies, key=lambda x: distance(self.position, x.position))
def choose_action(self):
    # choose between move, attack
    # if there is enemy nearby, attack
    # else move
    pass
def take_turn(self):
    action = self.choose_action(zombie)
    If the action is "attack":
        Loop through the surrounding area of the zombie
        If there is a human in the surrounding area:
            The zombie attacks the human
            If the human's health is less than or equal to zero:
                Remove the human from the list of humans
                Create a new zombie at the human's position
                Add the new zombie to the list of zombies
    If the action is "move":
        The zombie chooses a direction to move in
        Remove the zombie from the current position
        The zombie moves to the new position
        Add the zombie to the new position in the list of zombies.
random_move(self):
# check legal move
"""
"""
filter, zip, reduce
iterator, itertools
https://myapollo.com.tw/zh-tw/python-itertools-more-itertools/
assert
https://www.youtube.com/watch?v=96mDQrlceEk&ab_channel=Indently
metaprogramming (register lead classes, singleton)
https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Metaprogramming.html#the-metaclass-hook
function factory python
https://levelup.gitconnected.com/how-to-create-callable-objects-in-python-python-oop-complete-course-part-20-15fe46e3e2c3
use of hashing in game
Techniques for advanced functional programming using lambda functions and partial function application.
Concurrent programming methods to optimize how your code interacts with APIs.
Advanced control logic using iterators and generators.
"""

"""
unittest
https://github.com/Octavio-Velazquez/P0_Basic_Calculator/blob/main/test_operations.py
doctest
pytest
sys exit
debugging
use logging module logging.debug()/logging.info()/logging.warning()/logging.error()/logging.critical()
https://mp.weixin.qq.com/s?__biz=MzU4OTYzNjE2OQ==&mid=2247494272&idx=1&sn=adbc9770fc995785b061ace35f5a81c2&chksm=fdc8dda6cabf54b002dddd81c15a4eb45d5236561dfa823af0c33782a499449ea03cc72d07ac&scene=21#wechat_redirect
https://github.com/nedbat/coveragepy
https://ithelp.ithome.com.tw/articles/10078821
https://about.codecov.io/blog/writing-better-tests-with-ai-and-github-copilot/
Hypothesis
runtime type checking
typeguard
a debug mode that print all variables in each steps and allow user control of the agents in the map
https://pythonforundergradengineers.com/writing-tests-for-scientific-code.html
speed up
https://mp.weixin.qq.com/s/M7DdUWLzqOLVR7qJFmvZdA
concurrent.futures module
threading
https://python-course.eu/applications-python/threads.php
subprocessing
https://www.digitalocean.com/community/tutorials/how-to-use-subprocess-to-run-external-programs-in-python-3
https://codereview.stackexchange.com/questions/120790/distributed-system-simulator?rq=1
python package the project
dependency management
Pipenv is a direct competitor to Poetry, but I like Poetry better. Poetry is a more active project and works with the relatively new pyproject.toml file. I’m also a bit of a rebel, so having a group of Python developers who call themselves the “Python Packaging Authority” tell me to use Pipenv is a turn-off.
On the other hand, I have used pip-tools. If you aren’t ready to jump into a dependency manager like Poetry yet, but you would like to automate tracking transitive dependencies, pip-tools is a good start. The post RIP Pipenv has great arguments for using pip-tools. I would still use pip-tools, but I grew tired of writing pip-compile. Poetry makes everything easier.
logger decorator to specify when the function starts and ends
repeat decorator to repeat the function multiple times
retry decorator to retry if encounter an exception
countcall decorator to count the number of function calls
atexit register decorator to do something when the script is terminated
functools singledispatch decorator for function overloading
sonarqube
"""

# Advanced

# Here's an example of how you could design a Monte Carlo simulation of a zombie apocalypse:
# Define the population and their initial state: Start by defining the total population and their initial state, such as their location, health status (healthy or infected), and other relevant attributes. You could represent each individual in the population as an object in your simulation with properties such as location, health status, and decision-making capabilities.
# Model the spread of the zombie virus: Next, you'll need to model the spread of the zombie virus from infected individuals to healthy ones. This could be based on factors such as the proximity of healthy individuals to infected ones, the number of zombies in the area, and the likelihood of a healthy individual being bitten or coming into contact with infected fluids.
# Model the decision-making of survivors: You'll also need to model the decision-making of survivors as they try to evade zombies and find safety. This could be based on factors such as the proximity of safe havens, the number of zombies in the area, and the supplies they have on hand. You could use random numbers to model the uncertainty and randomness inherent in these decisions.
# Simulate the movement of the population: Next, you'll simulate the movement of the population over time. This could involve updating the location of each individual in the simulation based on their decisions and the factors affecting their movement, such as the presence of zombies and safe havens.
# Keep track of the state of the population: Keep track of the state of the population, including the number of healthy individuals, the number of infected individuals, and the number of fatalities. You could also keep track of other relevant metrics, such as the number of safe havens, the amount of supplies, and the distribution of the population over the landscape.
# Run the simulation multiple times: Finally, you'll want to run the simulation multiple times, perhaps hundreds or thousands of times, to generate a range of possible scenarios. This will provide valuable insights into the range of possible outcomes and help you prepare for a wide range of contingencies.
# Analyze the results: Once you have run the simulation multiple times, you can analyze the results to get a better understanding of the dynamics of the zombie apocalypse. For example, you could calculate statistics such as the average number of fatalities, the average number of survivors, and the average time to reach a safe haven. You could also generate graphs and visualizations to help you understand the results and identify any patterns or trends.

# Backtracking agent
# Monte carlo agent
# A* agent

# powerup and other resources and weapon may use same probability pickup function

# use one grid for item, one grid for weapon, one grid for other resources, one grid for human, one grid for zombie
# so that we can use 1 and 0 to represent whether there is an item or not and use numpy to do matrix operation
# control using underlying probabilistic model

# human may attack human

# cure to heal zombie to human or fight off infection else immediately turn into zombie

# Cellular Automata
# Conway's Game of Life
# fewer than 2 neighbors or more than 3 neighbors, die
# exactly neighbors, stay the same
# exactly 3 neighbors, become alive
# Predator-Prey
# surrounded by zombies, alive turn into zombie
# surrounded by humans, zombie dies

# smart weapon selection
# own a list of weapons with different damage and attack range
# if zombie is close, use melee weapon
# if zombie is far, use ranged weapon
# if zombie is far and no ranged weapon, use melee weapon
# if zombie is close and no melee weapon, use ranged weapon
# choose weapon according to distance and then damage
# each weapon has a different attack method (use strategy pattern)

# may move towards weapon and item inside visual range

# velocity and acceleration for human and zombie

# static equilibrium
# conservation of momentum

# speed controls who attacks first, if dead can't attack back
# or not in turn-based game, attack in interval of speed time after encountering (use threading)

# fire that can spread on the grid with time passing
# increment fire time by fire speed
# fire can hurt both human and zombie
# fire can be extinguished by water

# layout that affects the game
# add wall class, human and zombie can't move through wall, may place barrier to create complex grid
# add door class, only human can move through door
# add barricade class, human may place barricade which has health and can be destroyed by huamn and zombie

# show time step and health bar for human and zombie
# Hinton diagrams to visualize the health of the agent

# agents form a group and move together and attack, defend together

# https://replit.com/@ShiGame/TALKING-JOAQUIN#main.py
# https://zhuanlan.zhihu.com/p/134692590
# https://art-python.readthedocs.io/en/latest/

# A multi-objective problem related to a zombie apocalypse simulation could involve optimizing the allocation of resources to achieve several conflicting objectives simultaneously. For example, one objective could be to minimize the number of human casualties, while another objective could be to minimize the use of resources such as food, ammunition, and medical supplies.
# Another objective could be to maximize the number of survivors, while also minimizing the risk of infection and the spread of the zombie virus. A third objective could be to maximize the effectiveness of defense mechanisms, such as barricades and traps, while minimizing the time and effort required to maintain them.
# In this scenario, the optimization problem would involve finding a balance between these conflicting objectives and determining the optimal allocation of resources to achieve the best possible outcome under the given constraints. This type of problem is known as a multi-objective optimization problem, and it requires the use of advanced mathematical modeling techniques to generate solutions that are both feasible and efficient.

# Dynamic programming and recursive programming are both useful approaches for solving problems in computer science, but they have different strengths and weaknesses depending on the problem being tackled. In the context of a zombie apocalypse simulation, both approaches could be useful, but they would be applied in different ways.
# Dynamic programming is an algorithmic approach that solves a problem by breaking it down into smaller subproblems and storing the solutions to those subproblems in a table. By building up solutions to increasingly larger subproblems, dynamic programming can efficiently solve problems that would otherwise require repeated computations. In the context of a zombie apocalypse simulation, dynamic programming could be used to optimize survival strategies by breaking down the problem into smaller subproblems and building up a solution based on the optimal solutions to those subproblems. For example, one subproblem might be how to efficiently gather resources like food, water, and medicine, while another might be how to avoid or eliminate zombie threats. By solving these subproblems and storing the solutions, the simulation could optimize survival strategies for longer-term survival.
# Recursive programming, on the other hand, is an algorithmic approach that solves a problem by calling itself with smaller instances of the problem. Recursion is useful when a problem can be broken down into smaller instances of the same problem, and the base case can be easily identified. In the context of a zombie apocalypse simulation, recursive programming could be used to simulate the spread of the zombie virus or the movement of individual zombies. For example, a recursive function could be used to simulate the spread of the virus by calling itself with smaller instances of the infection until the entire population has been infected or the spread is contained.
# In summary, both dynamic programming and recursive programming could be useful in a zombie apocalypse simulation, but they would be applied in different ways. Dynamic programming would be useful for optimizing survival strategies over a longer time frame, while recursive programming would be useful for simulating the spread of the virus or the movement of individual zombies.

# Dynamic programming can be used for behaviour decision making, resources management, and pathfinding. It can include factors such as the risk of encountering zombies, the availability of resources, and the physical and mental state of the survivors.

# Recursive programming can be used in a zombie apocalypse simulation to model the spread of the zombie virus through a population. The recursive function would take an infected individual and then recursively call itself for each uninfected individual within a certain range. The range would be determined by the likelihood of infection and the distance between individuals. The function would then mark each newly infected individual as infected and call itself again for that individual.

# A hash map is a data structure that allows for efficient storage and retrieval of key-value pairs. You should use a hash map when you need to store data in a way that allows for fast lookup, insertion, and deletion of items based on a key.
# Here are some common scenarios where using a hash map can be particularly beneficial:
# When you have a large amount of data to store and quick lookups are necessary: Hash maps provide constant time complexity (O(1)) for lookups, which means that finding a value based on a key takes the same amount of time regardless of the size of the data set.
# When you need to maintain a collection of unique keys: Hash maps use unique keys to store and retrieve values, which makes them useful for maintaining a collection of unique items.
# When you need to associate one value with another: Hash maps store key-value pairs, which makes them a useful tool for associating one value with another.
# When you need to frequently update or modify data: Hash maps provide efficient insertions and deletions, making them a good choice for situations where data is frequently updated.
# When you need to perform operations that involve searching for specific values: Hash maps can be used to store and search for values based on specific criteria, such as finding all values that meet a certain condition.
# Overall, hash maps are a powerful tool for storing and accessing data in a way that is both efficient and flexible.
# In a zombie apocalypse simulation, you can use hash maps to store information about various entities in the game, such as:
# Map grid: You can create a hash map to store information about different locations on the map grid, such as buildings, roads, and open spaces. Each location can be assigned a unique key, and the value can be an object that stores information about the location, such as its type, status (zombie-infested or not), and the number of resources it contains.
# Zombies and humans: You can use a hash map to store information about the zombies and human characters in the game. Each character can be assigned a unique key, and the value can be an object that stores information about the character, such as its current location, health status, and inventory.
# Weapons and resources: You can use a hash map to store information about the weapons and resources available in the game. Each weapon or resource can be assigned a unique key, and the value can be an object that stores information about the item, such as its type, damage, and availability.
# Using hash maps in this way can make it easier and faster to perform certain operations in the simulation, such as searching for characters or items based on their unique key or updating the status of a location on the map grid. It can also help reduce the amount of memory and processing power required to store and manipulate large amounts of data in the simulation.

# There are several data structures that you can use in a zombie apocalypse simulation, depending on the specific needs of your simulation. Here are a few examples:
# Graphs: Graphs are useful for representing the connections between different locations in the simulation. You can use a graph to represent the layout of a city or town, with nodes representing buildings and edges representing roads or paths between them. You can use graph algorithms such as Dijkstra's algorithm or A* search to find the shortest path between two locations, which can be useful for simulating movement and travel.
# Queues: Queues can be used to represent lines of people waiting for resources or services. For example, you can use a queue to simulate the line of people waiting to get into a safe zone or to receive medical attention. You can use queue algorithms such as first-in-first-out (FIFO) or priority queues to manage the order in which people are served.
# Arrays: Arrays can be used to represent populations of people or zombies, with each element representing an individual. You can use arrays to keep track of a person's health status, location, and other attributes. You can use array algorithms such as sorting and searching to find specific individuals or groups within the population.
# Trees: Trees can be used to represent the hierarchy of leadership or organization within a group of survivors. For example, you can use a tree to represent a chain of command within a military organization or a group of survivors. You can use tree algorithms such as depth-first search or breadth-first search to traverse the tree and access information about the different levels of leadership.
# Stacks: Stacks can be used to represent resources that can be used up or consumed, such as food or ammunition. You can use a stack to keep track of the remaining resources and remove items as they are used up. You can use stack algorithms such as last-in-first-out (LIFO) to manage the order in which resources are consumed.
