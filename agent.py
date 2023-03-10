from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Union


class Agent(ABC):
    
    @abstractmethod
    def __init__(self, id: int, position: tuple[int, int], health: int=100, strength: int=10, armour: int=5, speed: int=1):
        self.id = id
        self.position = position
        self.health = health
        self.strength = strength
        self.armour = armour
        self.speed = speed
        
        # 0's and 1's to represent the agent's genome
        # the genome can be used to determine the agent's behavior
        # state, health and size can be used to determine the agent's fitness
        # location, direction, speed, energy, infection, infection time, death
        # can be used to determine the agent's state

    def move(self, direction):
        self.position = (self.position[0]+direction[0], 
                        self.position[1]+direction[1])
    
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
    
    @abstractmethod
    def scavenge(self, possible_weapons):
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
    def __init__(self, id, position, health, strength, armour, speed, intelligence, weapon=None):
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
            zombie = Zombie(len(apocalypse.zombie_manager.zombies), self.position, 100, 10, 0, 1)
            apocalypse.zombie_manager.add_agent(zombie)
            
    @property
    def defense(self):
        return self.armour*random.uniform(0.5, 1.5)
            
    """
    @cached_property
    def closest_enemy(self):
        return min(apocalypse.zombie_manager.zombies, key=lambda x: distance(self.position, x.position))
    """
    """
    def choose_action(self):
        # choose between move, attack, pick up item, use item
        # if low health, use item
        # elif there is enemy nearby, attack
        # elif there is item nearby, pick up
        # elif there is human nearby, trade
        # elif there is enough condition, craft item
        # else move
        pass
    """
    
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
            
    # move, then interact, then next turn
    """
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
            Remove the human from the current location
            The human moves to the new location
            Add the human to the new location in the list of humans.
    
    # choosing direction can go to a separate self.choose_direction(agent) method
    # attack can go to a separate self.attack_neighbors(agent) method
    """
    
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

# PowerUp is created randomly and found
# not created when found

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
                    
    # may trade weapon and armour
    # more clever conditions, based on intelligence
    # trade different items, like specific amount of food for weapon
    # craft method that takes food and turns it into item

    def print_human_info(self):
        print("Humans:")
        for human in self.humans:
            print(f"Human {human.id}: health={human.health}, position={human.position}, weapon={human.weapon}")


class Zombie(Agent):

    def __init__(self, id, position, health, strength, armour, speed):
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
            
    """
    @cached_property
    def closest_enemy(self):
        return min(apocalypse.zombie_manager.zombies, key=lambda x: distance(self.position, x.position))
    """
    """
    def choose_action(self):
        # choose between move, attack
        # if there is enemy nearby, attack
        # else move
        pass
    """
    
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
                
    """
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
            Remove the zombie from the current location
            The zombie moves to the new location
            Add the zombie to the new location in the list of zombies.
    """
    
    def random_move(self):
        # Move the zombie to a random position
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        self.move((dx, dy))
        
    # check legal move

    def attack(self, human):
        damage = self.strength*random.uniform(0.5, 1.5)
        # Deal 10 damage to the human
        human.take_damage(damage)
        
    def scavenge(self, possible_weapons):
        pass

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
    
    location: tuple[int, int]
    health: int
    strength: int
    armour: int
    speed: int
    intelligence: int
    
    def __str__(self):
        return f"PowerUp at {self.location}: health={self.health}, strength={self.strength}, armour={self.armour}, speed={self.speed}, intelligence={self.intelligence}"

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

# store counter in factory
# may use builder pattern if the product is composite or if there are more parameters

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
    
    def __str__(self):
        return "\n".join(["".join([str(cell) for cell in row]) for row in self.grid])
    
    def __getitem__(self, location):
        return self.grid[location[0]][location[1]]
    
    def __setitem__(self, location, value):
        self.grid[location[0]][location[1]] = value
        
    def get_human(self, location) -> Human:
        if isinstance(self.grid[location], Human):
            return self.grid[location[0]][location[1]]
        else:
            raise Exception('No human at this location')
        
    def get_zombie(self, location) -> Zombie:
        if isinstance(self.grid[location], Zombie):
            return self.grid[location[0]][location[1]]
        else:
            raise Exception('No zombie at this location')
        
    def get_item(self, location) -> Union[Weapon, PowerUp]:
        if isinstance(self.grid[location], Weapon) or isinstance(self.grid[location], PowerUp):
            return self.grid[location]
        else:
            raise Exception('No item at this location')

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
            "location": (random.randint(0, school_size-1),
                        random.randint(0, school_size-1))
        }
        return self.factory.produce(arguments)
        
    # ensure legal location
    # factory pattern for weapons
    # factory pattern for map
        
    def initialize(self, num_zombies, num_humans, school_size):
        # a 2D list representing the map, with None representing a empty cell and a human or zombie representing a cell occupied by human or zombie
        self.grid = Grid(school_size, school_size)
        # Initializes the humans and zombies in the map.
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
                    self.grid[human.position[0]][human.position[1]] = None
                    self.human_manager.humans.remove(human)
            for zombie in self.zombie_manager.zombies:
                if zombie.health <= 0:
                    self.grid[zombie.position[0]][zombie.position[1]] = None
                    self.zombie_manager.zombies.remove(zombie)
            # End the game if there are no more humans or zombies.
            if self.end_condition():
                break

    # move agent to a new location only if the new location is empty
    
    def escape(self, agent):
        # define a escape location for the humans to win
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

"""

"""
This revised code separates the simulation and its agents into separate classes. The ZombieApocalypse class stores instances of the HumanManager and ZombieManager classes, and has methods to advance the simulation by a single time step, add or remove humans and zombies from the simulation, and print a representation of the simulation map. The Human and Zombie classes both inherit from the Agent class and have additional attributes and methods specific to their roles in the simulation. The HumanManager and ZombieManager classes have methods to add and remove humans and zombies, respectively.
You can use this code to create and manipulate a simulation of a zombie apocalypse with humans and zombies. You can initialize the simulation with a certain number of humans and zombies using the ZombieApocalypse class, and advance the simulation by a single time step using its step method. You can add or remove humans and zombies from the simulation during runtime using the add_human and remove_human methods of the HumanManager class, and the add_zombie and remove_zombie methods of the ZombieManager class. You can also visualize the state of the simulation or retrieve information about the positions and attributes of the humans and zombies using the print_map method of the ZombieApocalypse class or the get_agents_within_range method of the Human or Zombie class.
"""

"""
map, filter, zip, reduce
namedtuple
functools
iterator, itertools
https://myapollo.com.tw/zh-tw/python-itertools-more-itertools/
collections
assert
https://www.youtube.com/watch?v=96mDQrlceEk&ab_channel=Indently
metaprogramming (register lead classes, singleton)
https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Metaprogramming.html#the-metaclass-hook
function factory python
https://levelup.gitconnected.com/how-to-create-callable-objects-in-python-python-oop-complete-course-part-20-15fe46e3e2c3
namedtuple vs dataclass
use of hashing in game
Techniques for advanced functional programming using lambda functions and partial function application.
Concurrent programming methods to optimize how your code interacts with APIs.
Advanced control logic using iterators and generators.
"""

"""
sys exit
debugging
use logging module logging.debug()/logging.info()/logging.warning()/logging.error()/logging.critical()
https://mp.weixin.qq.com/s?__biz=MzU4OTYzNjE2OQ==&mid=2247494272&idx=1&sn=adbc9770fc995785b061ace35f5a81c2&chksm=fdc8dda6cabf54b002dddd81c15a4eb45d5236561dfa823af0c33782a499449ea03cc72d07ac&scene=21#wechat_redirect
https://github.com/nedbat/coveragepy
https://ithelp.ithome.com.tw/articles/10078821
https://about.codecov.io/blog/writing-better-tests-with-ai-and-github-copilot/
Hypothesis
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
"""

# Advanced

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

# agents form a group and move together and attack, defend together

# https://replit.com/@ShiGame/TALKING-JOAQUIN#main.py
# https://zhuanlan.zhihu.com/p/134692590
