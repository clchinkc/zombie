import pytest
from population import Individual, Population, State


@pytest.fixture
def population():
    return Population(10, 20)

def test_init_population(population):
    assert len(population.agent_list) == 20
    for individual in population.agent_list:
        assert isinstance(individual, Individual)

def test_add_individual(population):
    new_individual = Individual(21, State.HEALTHY, (3,5))
    population.add_individual(new_individual)
    assert len(population.agent_list) == 21
    assert new_individual in population.agent_list

def test_remove_individual(population):
    individual = population.agent_list[0]
    population.remove_individual(individual)
    assert len(population.agent_list) == 19
    assert individual not in population.agent_list

def test_update_grid(population):
    old_locations = [individual.location for individual in population.agent_list]
    population.update_grid()
    new_locations = [individual.location for individual in population.agent_list]
    assert old_locations != new_locations

def test_update_state(population):
    # Test that the state changes after a certain number of updates
    old_states = [individual.state for individual in population.agent_list]
    for i in range(10):
        population.update_state()
    new_states = [individual.state for individual in population.agent_list]
    assert old_states != new_states

def test_update_population_metrics(population):
    # Test that the metrics change after a certain number of updates
    old_metrics = population.num_healthy, population.num_infected, population.num_zombie
    for i in range(10):
        population.update_state()
        population.update_population_metrics()
    new_metrics = population.num_healthy, population.num_infected, population.num_zombie
    assert old_metrics != new_metrics
    
if __name__ == "__main__":
    pytest.main(["-v", __file__])