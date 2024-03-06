from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
from individual import Individual
from school import School
from states import HealthState


class Population:
    def __init__(self, school_size: int, population_size: int) -> None:
        self.school: School = School(school_size)
        self.agent_list: list[Individual] = []
        self.severity: float = 0.0
        self.init_population(school_size, population_size)
        self.update_population_metrics()
        self.observers = []

    def add_individual(self, individual: Individual) -> None:
        self.agent_list.append(individual)
        self.school.add_individual(individual)

    def remove_individual(self, individual: Individual) -> None:
        self.agent_list.remove(individual)
        self.school.remove_individual(individual.location)

    def create_individual(self, id: int, school_size: int) -> Individual:
        state = random.choices(list(HealthState), weights=[0.7, 0.1, 0.2, 0.0])[0]
        available_locations = [(i, j) for i in range(school_size) for j in range(school_size) if self.school.legal_location((i, j))]
        location = random.choice(available_locations)
        return Individual(id, state, location)

    def init_population(self, school_size: int, population_size: int) -> None:
        for i in range(population_size):
            individual = self.create_individual(i, school_size)
            self.add_individual(individual)

    def clear_population(self) -> None:
        self.agent_list.clear()
        self.school.grid = np.full((self.school.size, self.school.size), None, dtype=object)

    # a method to init using a grid of "A", "I", "Z", "D"

    def run_population(self, num_time_steps: int) -> None:
        for time in range(num_time_steps):
            self.timestep = time + 1
            print("Time step: ", self.timestep)
            self.severity = time / num_time_steps
            print("Severity: ", self.severity)
            self.school.update_grid(self.agent_list, self.migration_probability)
            print("Updated Grid")
            self.school.update_connections()
            print("Updated Connections")
            self.update_individual_states()
            print("Updated State")
            self.update_population_metrics()
            print("Updated Population Metrics")
            individual_info = self.get_all_individual_info()
            print(individual_info)
            print("Got Individual Info")
            school_info = self.school.get_info()
            print(school_info)
            print("Got School Info")
            self.notify_observers()
            print("Notified Observers")
        self.clear_population()

    def update_individual_states(self) -> None:
        for individual in self.agent_list:
            individual.update_state(self.severity)
            if individual.health_state == HealthState.DEAD:
                self.school.remove_individual(individual.location)

    def update_population_metrics(self) -> None:
        self.calculate_state_counts()
        self.calculate_probabilities()

    def calculate_state_counts(self) -> None:
        state_counts = Counter([individual.health_state for individual in self.agent_list])
        self.num_healthy = state_counts[HealthState.HEALTHY]
        self.num_infected = state_counts[HealthState.INFECTED]
        self.num_zombie = state_counts[HealthState.ZOMBIE]
        self.num_dead = state_counts[HealthState.DEAD]
        self.population_size = self.num_healthy + self.num_infected + self.num_zombie
        
    def calculate_probabilities(self) -> None:
        self.infection_probability = 1 - (1 / (1 + math.exp(-self.severity))) # logistic function
        self.turning_probability = self.severity / (1 + self.severity) # softplus function
        self.death_probability = self.severity  # linear function
        self.migration_probability = self.population_size / (self.population_size + 1)

        # may use other metrics or functions to calculate the probability of infection, turning, death, migration

    def get_all_individual_info(self) -> str:
        return f"Population of size {self.population_size}\n" + \
            "\n".join(individual.get_info() for individual in self.agent_list)

    def attach_observer(self, observer: 'Observer') -> None:
        self.observers.append(observer)

    def notify_observers(self) -> None:
        for observer in self.observers:
            observer.update()

    def __str__(self) -> str:
        return f"Population with {self.num_healthy} healthy, {self.num_infected} infected, and {self.num_zombie} zombie individuals"

    def __repr__(self) -> str:
        return f"Population({self.school.size}, {self.population_size})"








