from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from individual import Individual
from states import HealthState

# Strategy pattern

class MovementStrategy(ABC):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]

    @abstractmethod
    def choose_direction(self, individual, school):
        raise NotImplementedError


@dataclass
# if no neighbors, choose random direction
class RandomMovementStrategy(MovementStrategy):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]

    def choose_direction(self):
        return random.choice(self.legal_directions)
    
@dataclass
# simulate Brownian motion
class BrownianMovementStrategy(MovementStrategy):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]
    
    def __post_init__(self):
        self.damping_coefficient = 0.5
        self.std_dev = 1
        self.scale_factor = math.sqrt(2 * self.damping_coefficient)*self.std_dev

    def choose_direction(self):
        while True:
            direction = np.rint(np.random.normal(loc=0, scale=self.scale_factor, size=2)).astype(int)
            if tuple(direction) in self.legal_directions:
                return direction

@dataclass
# if healthy or other having no alive neighbors
class FleeZombiesStrategy(MovementStrategy):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]

    def choose_direction(self):
        zombies_locations = [zombies.location for zombies in self.neighbors if zombies.health_state == HealthState.ZOMBIE]
        return self.direction_against_closest(self.individual, self.legal_directions, zombies_locations)

    # find the closest zombie and move away from it
    def direction_against_closest(self, individual: Individual, legal_directions: list[tuple[int, int]], target_locations: list[tuple[int, int]]) -> tuple[int, int]:
        distances = [np.linalg.norm(np.subtract(np.array(individual.location), np.array(target))) for target in target_locations]
        closest_index = np.argmin(distances)
        closest_target = target_locations[closest_index]
        direction_distances = [np.linalg.norm(np.add(d, np.array(individual.location)) - closest_target) for d in np.array(legal_directions)]
        max_distance = np.max(direction_distances)
        farthest_directions = np.where(direction_distances == max_distance)[0]
        return legal_directions[random.choice(farthest_directions)]

@dataclass
# if zombie or other having no zombie neighbors
class ChaseHumansStrategy(MovementStrategy):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]

    def choose_direction(self):
        alive_locations = [alive.location for alive in self.neighbors if alive.health_state == HealthState.HEALTHY]
        return self.direction_towards_closest(self.individual, self.legal_directions, alive_locations)

    # find the closest human and move towards it
    def direction_towards_closest(self, individual: Individual,legal_directions: list[tuple[int, int]],target_locations: list[tuple[int, int]],) -> tuple[int, int]:
        current_location = np.array(individual.location)
        new_locations = current_location + np.array(legal_directions)
        distance_matrix = np.linalg.norm(new_locations[:, np.newaxis, :] - np.array(target_locations)[np.newaxis, :, :], axis=2)
        min_distance = np.min(distance_matrix[distance_matrix != 0])
        min_directions = np.where(distance_matrix == min_distance)[0]
        return legal_directions[random.choice(min_directions)]

    # If the closest person is closer than the closest zombie, move towards the person, otherwise move away from the zombie
    # or move away from zombie is the priority and move towards person is the secondary priority


@dataclass
class NoMovementStrategy(MovementStrategy):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]

    def choose_direction(self):
        return (0, 0)


# may use init to store individual and school
# may use function, use closure to store individual and school


class MovementStrategyFactory:
    def create_strategy(self, individual, school):
        # get legal directions
        legal_directions = school.get_legal_directions(individual)
        # early exit
        if not legal_directions:
            return NoMovementStrategy(individual, legal_directions, [])
        # get neighbors
        neighbors = school.get_neighbors(individual.location, individual.sight_range)
        # early exit
        # if no neighbors, random movement
        if not neighbors:
            return RandomMovementStrategy(individual, legal_directions, neighbors)
        # get number of human and zombies neighbors
        alive_number = len([alive for alive in neighbors if alive.health_state == HealthState.HEALTHY])
        zombies_number = len([zombies for zombies in neighbors if zombies.health_state == HealthState.ZOMBIE])
        # if no human neighbors, move away from the closest zombies
        if alive_number == 0 and zombies_number > 0:
            return FleeZombiesStrategy(individual, legal_directions, neighbors)
        # if no zombies neighbors, move towards the closest human
        elif zombies_number == 0 and alive_number > 0:
            return ChaseHumansStrategy(individual, legal_directions, neighbors)
        # if both human and zombies neighbors, zombies move towards the closest human and human move away from the closest zombies
        else:
            if individual.health_state == HealthState.ZOMBIE and alive_number > 0:
                return ChaseHumansStrategy(individual, legal_directions, neighbors)
            elif (individual.health_state == HealthState.HEALTHY or individual.health_state == HealthState.INFECTED) and zombies_number > 0:
                return FleeZombiesStrategy(individual, legal_directions, neighbors)
            elif individual.health_state == HealthState.DEAD:
                return NoMovementStrategy(individual, legal_directions, neighbors)
            else:
                return BrownianMovementStrategy(individual, legal_directions, neighbors)

    # may consider update the grid according to the individual's location
    # after assigning all new locations to the individuals
    # may add a extra attribute to store new location
