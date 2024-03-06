from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from enum import Enum, auto

from states import HealthState

# State pattern

class StateMachine(ABC):
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @abstractmethod
    def update_state(self, individual: "Individual", severity: float) -> None:
        pass

    def is_infected(self, individual: "Individual", severity: float, randomness=random.random()) -> bool:
        infection_probability = 1 / (1 + math.exp(-severity))
        if any(other.health_state == HealthState.ZOMBIE for other in individual.connections):
            if randomness < infection_probability:
                return True
        return False

    def is_turned(self, individual: "Individual", severity: float, randomness=random.random()) -> bool:
        turning_probability = individual.infection_severity
        if randomness < turning_probability:
            return True
        return False

    def is_died(self, individual: "Individual", severity: float, randomness=random.random()) -> bool:
        death_probability = severity
        if any(other.health_state == HealthState.HEALTHY or other.health_state == HealthState.INFECTED for other in individual.connections):
            if randomness < death_probability:
                return True
        return False

# can add methods that change behaviour based on the state

class HealthyStateMachine(StateMachine):
    def update_state(self, individual: "Individual", severity: float) -> None:
        if self.is_infected(individual, severity):
            individual.health_state = HealthState.INFECTED

class InfectedStateMachine(StateMachine):
    def update_state(self, individual: "Individual", severity: float) -> None:
        individual.infection_severity = round(min(1, individual.infection_severity + 0.1), 1)
        if self.is_turned(individual, severity):
            individual.health_state = HealthState.ZOMBIE
        elif self.is_died(individual, severity):
            individual.health_state = HealthState.DEAD

class ZombieStateMachine(StateMachine):
    def update_state(self, individual: "Individual", severity: float) -> None:
        if self.is_died(individual, severity):
            individual.health_state = HealthState.DEAD

class DeadStateMachine(StateMachine):
    def update_state(self, individual: "Individual", severity: float) -> None:
        pass

class StateMachineFactory:
    @staticmethod
    def get_instance():
        return {
            HealthState.HEALTHY: HealthyStateMachine(),
            HealthState.INFECTED: InfectedStateMachine(),
            HealthState.ZOMBIE: ZombieStateMachine(),
            HealthState.DEAD: DeadStateMachine()
        }

    @staticmethod
    def update_state(individual: "Individual", severity: float) -> None:
        state_machines = StateMachineFactory.get_instance()
        state_machine = state_machines[individual.health_state]
        state_machine.update_state(individual, severity)
