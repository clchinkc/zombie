from __future__ import annotations
import time
import random
import math
from enum import Enum, auto
import numpy as np
from timeit_decorator import performance_decorator

# Define the states and transitions for the state machine model
class State(Enum):
    
    HEALTHY = auto()
    INFECTED = auto()
    ZOMBIE = auto()
    DEAD = auto()
    
    @classmethod
    def name_list(cls) -> list[str]:
        return [enm.name for enm in State]
    
    @classmethod
    def value_list(cls) -> list[int]:
        return [enm.value for enm in State]

class Agent:

    __slots__ = "id", "state", "location", "connections", \
    "infection_severity", "interact_range", "sight_range", "__dict__"
    
    def __init__(self, id: int, state: State, location: tuple[int, int]) -> None:
        self.id: int = id
        self.state: State = state
        self.location: tuple[int, int] = location
        self.connections: list[Agent] = []
        self.infection_severity: float = 0.0
        self.interact_range: int = 2
        self.sight_range: int = 5
        
        # different range for different states
        # may use random distribution

    def add_connection(self, other: Agent) -> None:
        self.connections.append(other)

    def move(self, direction: tuple[int, int]) -> None:
        self.location = tuple(np.add(self.location, direction))
        # self.location[0] += direction[0]
        # self.location[1] += direction[1]

    def update_state(self, severity: float) -> None:
        # Update the state of the agent based on the current state and the interactions with other people
        if self.state == State.HEALTHY:
            if self.is_infected(severity):
                self.state = State.INFECTED
        elif self.state == State.INFECTED:
            self.infection_severity += 0.1
            if self.is_turned():
                self.state = State.ZOMBIE
            elif self.is_died(severity):
                self.state = State.DEAD
        elif self.state == State.ZOMBIE:
            if self.is_died(severity):
                self.state = State.DEAD

    # cellular automaton
    def is_infected(self, severity: float) -> bool:
        infection_probability = 1 / (1 + math.exp(-severity))
        for agent in self.connections:
            if agent.state == State.ZOMBIE:
                if random.random() < infection_probability:
                    return True
        return False

    def is_infected_v2(self, severity: float) -> bool:
        infection_probability = 1 / (1 + math.exp(-severity))
        infectious = [agent.state == State.ZOMBIE and random.random() < infection_probability for agent in self.connections]
        return any(infectious)
    
    # cellular automaton
    def is_turned(self) -> bool:
        turning_probability = self.infection_severity
        if random.random() < turning_probability:
            return True
        return False

    # cellular automaton
    def is_died(self, severity: float) -> bool:
        death_probability = severity
        num_alive = sum(1 for agent in self.connections if agent.state == State.HEALTHY or agent.state == State.INFECTED)
        if random.random() < (1-(1-death_probability)**num_alive):
            return True
        return False
    
    def get_info(self) -> str:
        return f"Agent {self.id} is {self.state.name} and is located at {self.location}, having connections with {self.connections}, infection severity {self.infection_severity}, interact range {self.interact_range}, and sight range {self.sight_range}."

    def __str__(self) -> str:
        return f"Agent {self.id}"
    
    def __repr__(self) -> str:
        return "%s(%d,%d, %d)" % (self.__class__.__name__, self.id, self.state.value, self.location)

# use performance decorator to test the performance of the two versions of the code



def test_performance_1():
    connections = [Agent(i, State.ZOMBIE, (random.randint(0, 100), random.randint(0, 100))) for i in range(10000)]

    # Test the first version of the code
    start_time = time.time()
    for i in range(10000):
        connections[i].is_infected(0.9)
    end_time = time.time()
    print(f"First version took {end_time - start_time} seconds")

    connections = [Agent(i, State.ZOMBIE, (random.randint(0, 100), random.randint(0, 100))) for i in range(10000)]

    # Test the second version of the code
    start_time = time.time()
    for i in range(10000):
        connections[i].is_infected_v2(0.9)
    end_time = time.time()
    print(f"Second version took {end_time - start_time} seconds")

# test_performance_1()

from population import *

def test_performance_2():
    population = Population(100, 1)
    # add 1 individuals to the school
    individual = population.create_individual(2, 100)
    population.add_individual(individual)
    # Test the first version of the code
    start_time = time.time()
    for i in range(1000000):
        population.school.random_move(individual)
    end_time = time.time()
    print(f"First version took {end_time - start_time} seconds")

    # Test the second version of the code
    start_time = time.time()
    for i in range(1000000):
        population.school.random_move(individual)
    end_time = time.time()
    print(f"Second version took {end_time - start_time} seconds")
    
# test_performance_2()