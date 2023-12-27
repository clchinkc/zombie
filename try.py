
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional, Protocol, Union


class Position(NamedTuple):
    x: int
    y: int


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
    
    def __init__(self):
        self.agents = []

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)
    
    def remove_agent(self, agent: Agent) -> None:
        self.agents.remove(agent)

    def get_agents_in_range(self, agent: Agent, range: float):
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
        _, agents_in_range = self.get_agents_in_range(agent, range)
        self.connect_agents(agent, agents_in_range)

        # zombie may continue following a human even if other humans are closer

# Define the HumanProtocol
class HumanProtocol(Protocol):
    intelligence: int
    inventory: dict[str, Union['Weapon', 'PowerUp']]
    weapon: Union['Weapon', None]
    connections: list[Agent]

    @abstractmethod
    def scavenge(self, possible_weapons: list['Weapon']):
        """Method for scavenging weapons or supplies."""
        pass

    @abstractmethod
    def use_item(self, item: Union['Weapon', 'PowerUp']):
        """Method for using an item from the inventory."""
        pass

    @abstractmethod
    def attack(self, zombie: 'Zombie'):
        """Method for attacking a zombie."""
        pass

# Define the ZombieProtocol
class ZombieProtocol(Protocol):
    connections: list[Agent]

    @abstractmethod
    def attack(self, human: 'Human'):
        """Method for attacking a human."""
        pass

class Human(Agent, HumanProtocol):
    def __init__(self, id: int, position: Position, health: int, strength: int, armour: int, speed: int, intelligence: int, weapon: Union[Weapon, None]=None):
        super().__init__(id, position, health, strength, armour, speed)
        self.intelligence = intelligence
        self.inventory = {}
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
        return super().take_damage(max(0, damage - self.armour))
        
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
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        new_x = max(0, min(self.position.x + dx, apocalypse.grid.width - 1))
        new_y = max(0, min(self.position.y + dy, apocalypse.grid.height - 1))
        self.move((new_x, new_y))
        
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
            self.inventory['supplies'] = supplies
        # Roll a dice to determine if the human finds a new weapon
        if random.random() < 0.2:
            # Human has found a new weapon
            new_weapon = random.choice(possible_weapons)
            if self.weapon is None or (new_weapon.damage * new_weapon.range > self.weapon.damage * self.weapon.range):
                self.weapon = new_weapon

    def use_item(self, item):
        if item in self.inventory and isinstance(item, PowerUp):
            self.health += item.health
            self.strength += item.strength
            self.armour += item.armour
            self.speed += item.speed
            self.intelligence += item.intelligence
            self.inventory.pop(item)
        else:
            print(f"Item {item} not in inventory")


# Manage all humans in the apocalypse
class HumanManager(AgentManager):
    
    def __init__(self):
        super().__init__()
        self.humans = []
        
    def add_agent(self, human):
        self.agents.append(human)
        self.humans.append(human)
        
    def remove_agent(self, human):
        self.agents.remove(human)
        self.humans.remove(human)
        
    def get_agents_in_range(self, agent: Agent, range: float):
        return super().get_agents_in_range(agent, range)

    def get_enemies_in_attack_range(self, agent: Human):
        attack_range = math.sqrt(2) + (agent.weapon.range if agent.weapon else 0)
        agents_in_range = super().get_agents_in_range(agent, attack_range)
        enemies_in_range = [(distance, enemy) for distance, enemy in agents_in_range if isinstance(enemy, Zombie)]
        return enemies_in_range
    
    def get_closest_zombie(self, agent: Human):
        enemies = self.get_enemies_in_attack_range(agent)
        if len(enemies) == 0:
            return None
        closest_enemy = enemies[0]
        return closest_enemy
        
    def trade_with(self, agent1, agent2):
        if agent1.weapon and agent2.weapon:
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


class Zombie(Agent, ZombieProtocol):

    def __init__(self, id: int, position: Position, health: int, strength: int, armour: int, speed: int):
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
        return super().take_damage(max(0, damage - self.armour))

    def take_turn(self, closest_human):
        # If there are any humans in range, attack the closest one
        if closest_human is not None:
            if random.random() < 0.5:
                self.attack(closest_human)
            else:
                self.random_move()
    
    def random_move(self):
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        new_x = max(0, min(self.position.x + dx, apocalypse.grid.width - 1))
        new_y = max(0, min(self.position.y + dy, apocalypse.grid.height - 1))
        self.move((new_x, new_y))

    def attack(self, human):
        damage = self.strength*random.uniform(0.5, 1.5)
        # Deal 10 damage to the human
        human.take_damage(damage)

# Manage all zombies in the apocalypse
class ZombieManager(AgentManager):

    def __init__(self):
        super().__init__()
        self.zombies = []

    def add_agent(self, zombie):
        self.agents.append(zombie)
        self.zombies.append(zombie)

    def remove_agent(self, zombie):
        self.agents.remove(zombie)
        self.zombies.remove(zombie)

    def get_enemies_in_attack_range(self, agent: Zombie):
        attack_range = math.sqrt(2)  # Assuming zombies have a fixed attack range
        agents_in_range = super().get_agents_in_range(agent, attack_range)
        enemies_in_range = [(distance, enemy) for distance, enemy in agents_in_range if isinstance(enemy, Human)]
        return enemies_in_range
    
    def get_closest_human(self, agent: Zombie):
        enemies = self.get_enemies_in_attack_range(agent)
        if len(enemies) == 0:
            return None
        closest_enemy = enemies[0][1]  # Getting the human agent from the tuple
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

# Builder Class
class AgentBuilder:
    def __init__(self, agent_constructor):
        self.agent_constructor = agent_constructor
        self.params = {}

    def set_id(self, id: int):
        self.params['id'] = id
        return self

    def set_position(self, position: Position):
        self.params['position'] = position
        return self

    def set_health(self, health: int):
        self.params['health'] = health
        return self

    def set_strength(self, strength: int):
        self.params['strength'] = strength
        return self

    def set_armour(self, armour: int):
        self.params['armour'] = armour
        return self

    def set_speed(self, speed: int):
        self.params['speed'] = speed
        return self

    def set_intelligence(self, intelligence: int):
        self.params['intelligence'] = intelligence
        return self

    def set_weapon(self, weapon: Optional[Weapon]):
        self.params['weapon'] = weapon
        return self

    def build(self):
        agent = self.agent_constructor(**self.params)
        self.params = {}
        return agent


# Abstract Factory Class using Builder
class AbstractAgentFactory(ABC):
    @abstractmethod
    def create_agent(self, builder: AgentBuilder, **kwargs) -> Agent:
        raise NotImplementedError()


# Concrete Factory for Humans
class HumanFactory(AbstractAgentFactory):
    id_counter = 0

    def create_agent(self, **kwargs) -> Human:
        HumanFactory.id_counter += 1  # Increment the ID counter
        builder = AgentBuilder(Human)
        return builder.set_id(HumanFactory.id_counter) \
                      .set_position(kwargs.get("position", Position(0, 0))) \
                      .set_health(kwargs.get("health", 100)) \
                      .set_strength(kwargs.get("strength", 10)) \
                      .set_armour(kwargs.get("armour", 5)) \
                      .set_speed(kwargs.get("speed", 1)) \
                      .set_intelligence(kwargs.get("intelligence", 10)) \
                      .set_weapon(kwargs.get("weapon", None)) \
                      .build()

# Concrete Factory for Zombies
class ZombieFactory(AbstractAgentFactory):
    id_counter = 0

    def create_agent(self, **kwargs) -> Zombie:
        ZombieFactory.id_counter += 1  # Increment the ID counter
        builder = AgentBuilder(Zombie)
        return builder.set_id(ZombieFactory.id_counter) \
                      .set_position(kwargs.get("position", Position(0, 0))) \
                      .set_health(kwargs.get("health", 100)) \
                      .set_strength(kwargs.get("strength", 10)) \
                      .set_armour(kwargs.get("armour", 5)) \
                      .set_speed(kwargs.get("speed", 1)) \
                      .build()



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

    def produce(self, character_type: str, arguments: dict[str, Any]) -> Agent:
        """Create a game character of a specific type."""
        try:
            creator_func = self.character_creation_funcs[character_type]
        except KeyError:
            raise ValueError(f"unknown character type {character_type!r}") from None
        return creator_func.create_agent(arguments)


# Dataclass for Grid Cell
@dataclass
class Cell:
    content: Union[Human, Zombie, Weapon, PowerUp, None]

class Grid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[Cell(None) for _ in range(width)] for _ in range(height)]

    def __str__(self) -> str:
        grid_representation = ""
        for row in self.grid:
            for cell in row:
                if cell.content is None:
                    grid_representation += " . "
                elif isinstance(cell.content, Human):
                    grid_representation += " H "
                elif isinstance(cell.content, Zombie):
                    grid_representation += " Z "
                else:
                    grid_representation += " ? "
            grid_representation += "\n"
        return grid_representation

    def get_human(self, position: Position) -> Union[Human, None]:
        cell = self.grid[position.x][position.y]
        return cell.content if isinstance(cell.content, Human) else None

    def get_zombie(self, position: Position) -> Union[Zombie, None]:
        cell = self.grid[position.x][position.y]
        return cell.content if isinstance(cell.content, Zombie) else None

    def get_item(self, position: Position) -> Union[Weapon, PowerUp, None]:
        cell = self.grid[position.x][position.y]
        return cell.content if isinstance(cell.content, (Weapon, PowerUp)) else None

    def place_object(self, position: Position, obj: Union[Human, Zombie, Weapon, PowerUp, None]):
        self.grid[position.x][position.y].content = obj

    def remove_object(self, position: Position):
        self.grid[position.x][position.y].content = None

    def is_empty(self, position: Position) -> bool:
        return self.grid[position.x][position.y].content is None

    def print_map(self):
        print(self.__str__())


# Template method pattern
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
        self.grid.place_object(human.position, human)
        self.human_manager.add_agent(human)
        
    def remove_human(self, agent):
        self.grid.remove_object(agent.position)
        self.human_manager.remove_agent(agent)
        
    def add_zombie(self, zombie):
        self.grid.place_object(zombie.position, zombie)
        self.zombie_manager.add_agent(zombie)
        
    def remove_zombie(self, zombie):
        self.grid.remove_object(zombie.position)
        self.zombie_manager.remove_agent(zombie)
        
    def add_item(self, item):
        self.grid.place_object(item.position, item)
        
    def remove_item(self, item):
        self.grid.remove_object(item.position)
    
    def simulate(self, num_zombies, num_humans, school_size, num_turns):
        # Simulates the game.
        self.initialize(num_zombies, num_humans, school_size)
        self.run(num_turns)
        self.human_manager.print_human_info()
        self.zombie_manager.print_zombie_info()
        self.grid.print_map()
    
    # separate the creation and the use for humans and zombies
    def create_agent(self, school_size: int, type: str) -> Agent:
        arguments = {
            "position": Position(random.randint(0, school_size-1), random.randint(0, school_size-1))
        }
        return self.factory.produce(type, arguments)
        
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
            self.human_manager.add_agent(self.create_agent(school_size, "human"))
        for i in range(num_zombies):
            self.zombie_manager.add_agent(self.create_agent(school_size, "zombie"))
    
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
                    self.grid.remove_object(human.position)
                    self.human_manager.humans.remove(human)
            for zombie in self.zombie_manager.zombies:
                if zombie.health <= 0:
                    self.grid.remove_object(zombie.position)
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