from population import *
import pytest

# pytest for State
def test_state():
    assert State.name_list() == ["HEALTHY", "INFECTED", "ZOMBIE", "DEAD"]
    assert State.value_list() == [1, 2, 3, 4]
    
# pytest for Individual
def test_individual():
    individual = Individual(1, State.HEALTHY, (0, 0))
    assert individual.id == 1
    assert individual.state == State.HEALTHY
    assert individual.location == (0, 0)
    assert individual.connections == []
    assert individual.infection_severity == 0.0
    assert individual.interact_range == 2
    assert individual.sight_range == 5
    assert individual.get_info() == "Individual 1 is HEALTHY and is located at (0, 0), having connections with [], infection severity 0.0, interact range 2, and sight range 5."
    assert str(individual) == "Individual 1"
    # test behaviour
    individual2 = Individual(2, State.HEALTHY, (0, 1))
    individual.add_connection(individual2)
    assert individual.connections == [individual2]
    individual.move((1, 1))
    assert individual.location == (1, 1)
    """
    # test update_state
    individual.state = State.HEALTHY
    individual.update_state(100)
    assert individual.state == State.INFECTED
    for _ in range(10):
        individual.update_state(0)
    assert individual.state == State.ZOMBIE
    individual.update_state(10)
    assert individual.state == State.DEAD
    # test is_infected
    individual.state = State.HEALTHY
    individual3 = Individual(3, State.HEALTHY, (0, 2))
    individual.add_connection(individual3)
    assert individual.is_infected(10) == False
    """
    
def test_individual_class():
    indiv = Individual(1, State.HEALTHY, (0, 0))
    assert indiv.id == 1
    assert indiv.state == State.HEALTHY
    assert indiv.location == (0, 0)
    assert indiv.connections == []
    assert indiv.infection_severity == 0.0
    assert indiv.interact_range == 2
    assert indiv.sight_range == 5
    
    indiv.add_connection(Individual(2, State.INFECTED, (1, 1)))
    assert len(indiv.connections) == 1
    assert indiv.connections[0].id == 2
    
    indiv.move((1,1))
    assert indiv.location == (1,1)
    
    assert indiv.get_info() == "Individual 1 is HEALTHY and is located at (1, 1), having connections with [Individual(2, 2, (1, 1))], infection severity 0.0, interact range 2, and sight range 5."
    assert str(indiv) == "Individual 1"
    assert repr(indiv) == "Individual(1, 1, (1, 1))"
    
def test_update_state():
    random.seed(0)
    indiv = Individual(1, State.HEALTHY, (1, 1))
    indiv.add_connection(Individual(2, State.ZOMBIE, (0, 0)))
    indiv.add_connection(Individual(3, State.ZOMBIE, (0, 1)))
    indiv.add_connection(Individual(4, State.ZOMBIE, (0, 2)))
    indiv.add_connection(Individual(5, State.ZOMBIE, (1, 0)))
    indiv.add_connection(Individual(6, State.ZOMBIE, (1, 2)))
    indiv.add_connection(Individual(7, State.ZOMBIE, (2, 0)))
    indiv.add_connection(Individual(8, State.ZOMBIE, (2, 2)))
    indiv.update_state(1.0)
    assert indiv.state == State.INFECTED
    indiv.update_state(0.0)
    assert indiv.infection_severity == 0.1
    indiv.infection_severity = 1.0
    indiv.update_state(0.0)
    assert indiv.state == State.ZOMBIE
    indiv.connections = [Individual(2, State.HEALTHY, (0, 0))]
    indiv.update_state(1.0)
    assert indiv.state == State.DEAD

    
def test_is_died():
    indiv = Individual(1, State.INFECTED, (0, 0))
    indiv.connections = [Individual(2, State.ZOMBIE, (1, 1))]
    assert indiv.is_died(0.5) == False
    indiv.connections = [Individual(2, State.HEALTHY, (1, 1))]
    assert indiv.is_died(1) == True
    
def test_is_turned():
    indiv = Individual(1, State.INFECTED, (0, 0))
    indiv.infection_severity = 0.1
    assert indiv.is_turned() == False
    indiv.infection_severity = 1
    assert indiv.is_turned() == True

def test_is_infected():
    indiv = Individual(1, State.HEALTHY, (0, 0))
    indiv.connections = [Individual(2, State.HEALTHY, (1, 1))]
    assert indiv.is_infected(1.0) == False
    indiv.connections = [Individual(2, State.ZOMBIE, (1, 1))]
    assert indiv.is_infected(1.0) == True
