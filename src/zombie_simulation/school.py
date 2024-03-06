from __future__ import annotations

import random
from itertools import product
from typing import Optional

import numpy as np
from individual import Individual
from states import HealthState
from strategies import MovementStrategyFactory


class School:

    __slots__ = ("size", "grid", "strategy_factory", "__dict__",)

    def __init__(self, size: int) -> None:
        self.size = size
        # Create a 2D grid representing the school with each cell can contain a Individual object
        self.grid: np.ndarray = np.full((size, size), None, dtype=object)
        self.strategy_factory = MovementStrategyFactory()

        # may turn to width and height
        # may put Cell class in the grid where Cell class has individual attributes and rates

    def add_individual(self, individual: Individual) -> None:
        self.grid[int(individual.location[0])][int(
            individual.location[1])] = individual

    def get_individual(self, location: tuple[int, int]) -> Optional[Individual]:
        return self.grid[location[0]][location[1]]

    def remove_individual(self, location: tuple[int, int]) -> None:
        self.grid[location[0]][location[1]] = None

    # may change the add\remove function to accept individual class as well as location

    def update_connections(self) -> None:
        for row in self.grid:
            for cell in row:
                if cell is None:
                    continue
                neighbors = self.get_neighbors((cell.location), cell.interact_range)
                for neighbor in neighbors:
                    cell.add_connection(neighbor)

    def update_individual_location(self, individual: Individual) -> None:
        movement_strategy = self.strategy_factory.create_strategy(individual, self)
        direction = individual.choose_direction(movement_strategy)
        self.move_individual(individual, direction)

    # update the grid in the population based on their interactions with other people
    def update_grid(self, population: list[Individual], migration_probability: float, randomness=random.random()) -> None:
        for individuals in population:
            cell = self.get_individual(individuals.location)

            if cell is None or cell.health_state == HealthState.DEAD:
                continue

            if randomness < migration_probability:
                self.update_individual_location(cell)

                # if right next to then don't move
                # get neighbors with larger and larger range until there is a human
                # sight range is different from interact range

    def get_neighbors(self, location: tuple[int, int], interact_range: int = 2):
        x, y = location
        x_range = range(max(0, x - interact_range), min(self.size, x + interact_range + 1))
        y_range = range(max(0, y - interact_range), min(self.size, y + interact_range + 1))
        
        neighbors = filter(
            lambda pos: (pos[0] != x or pos[1] != y) and self.within_distance(self.grid[x][y], self.grid[pos[0]][pos[1]], interact_range),
            product(x_range, y_range)
        )

        return [self.grid[i][j] for i, j in neighbors]

    def within_distance(self, individual1: Optional[Individual], individual2: Optional[Individual], interact_range: int):
        if individual1 is None or individual2 is None:
            return False
        # check if the two individuals are within a certain distance of each other
        distance = self.distance(individual1.location, individual2.location)
        return distance < interact_range

    def distance(self, location1: tuple[int, int], location2: tuple[int, int]) -> float:
        # get the distance between two individuals
        distance = float(np.linalg.norm(np.subtract(location1, location2)))
        return distance

    def get_legal_directions(self, individual: Individual) -> list[tuple[int, int]]:
        # get all possible legal moves for the individual
        legal_directions = [
            (i, j) for i in range(-1, 2) for j in range(-1, 2)
            if (i == 0 and j == 0)
            or self.legal_location((individual.location[0] + i, individual.location[1] + j))
        ]
        return legal_directions

    def legal_location(self, location: tuple[int, int]) -> bool:
        return self.in_bounds(location) and self.not_occupied(location)

    def in_bounds(self, location: tuple[int, int]) -> bool:
        # check if the location is in the grid
        return (0 <= location[0] < self.size and 0 <= location[1] < self.size)

    def not_occupied(self, location: tuple[int, int]) -> bool:
        # check if the location is empty
        return self.grid[location[0]][location[1]] is None

    # return any(agent.position == (x, y) for agent in self.agents)

    def move_individual(self, individual: Individual, direction: tuple[int, int]) -> None:
        old_location = individual.location
        individual.move(direction)
        self.remove_individual(old_location)
        self.add_individual(individual)

    def get_info(self) -> str:
        return "\n".join(
            " ".join(str(individual.health_state.value) if individual else " " for individual in column)
            for column in self.grid
            )

    # return the count inside the grid
    
    def __str__(self) -> str:
        return f"School({self.size})"

    def __repr__(self) -> str:
        return "%s(%d,%d)" % (self.__class__.__name__, self.size)
