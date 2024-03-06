from __future__ import annotations

from enum import Enum, auto


class HealthState(Enum):

    HEALTHY = auto()
    INFECTED = auto()
    ZOMBIE = auto()
    DEAD = auto()

    @classmethod
    def name_list(cls) -> list[str]:
        return [enm.name for enm in HealthState]

    @classmethod
    def value_list(cls) -> list[int]:
        return [enm.value for enm in HealthState]

