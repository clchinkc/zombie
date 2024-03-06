from __future__ import annotations

from functools import cached_property

from states import HealthState
from state_machine import StateMachineFactory


class Individual:

    __slots__ = ("id","state","location","connections","infection_severity","interact_range","__dict__",)

    def __init__(self,id: int, health_state: HealthState, location: tuple[int, int],) -> None:
        self.id: int = id
        self.health_state: HealthState = health_state
        self.location: tuple[int, int] = location
        self.connections: list[Individual] = []
        self.infection_severity: float = 0.0
        self.interact_range: int = 2

        # different range for different states
        # may use random distribution

    @cached_property
    def sight_range(self) -> int:
        return self.interact_range + 3

    # fluent interface
    def add_connection(self, other: Individual) -> None:
        self.connections.append(other)

    def move(self, direction: tuple[int, int]) -> None:
        dx, dy = direction
        x, y = self.location
        self.location = (x + dx, y + dy)

    def choose_direction(self, movement_strategy) -> tuple[int, int]:
        return movement_strategy.choose_direction()

    def update_state(self, severity: float) -> None:
        StateMachineFactory.update_state(self, severity)

    def get_info(self) -> str:
        return f"Individual {self.id} is {self.health_state.name} and is located at {self.location}, having connections with {self.connections}, infection severity {self.infection_severity}, interact range {self.interact_range}, and sight range {self.sight_range}."

    def __str__(self) -> str:
        return f"Individual {self.id}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.id}, {self.health_state}, {self.location})"


# separate inheritance for human and zombie class
