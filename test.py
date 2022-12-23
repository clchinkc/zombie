"""
Implementing a simulation to model a zombie apocalypse at a school in Python would involve creating a program that uses the population model, state machine model, and cellular automaton to represent the size, behavior, and evolution of the school's population over time. Here is one possible way to implement this simulation in Python:
"""


import math
import random
import matplotlib.pyplot as plt

# Define the states and transitions for the state machine model
STATES = ["alive", "infected", "zombie", "dead"]


"""
TRANSITIONS = {
    "infection": ["alive", "infected"],
    "turning": ["infected", "zombie"],
    "death": ["zombie", "dead"]
}
"""
class Person:
    def __init__(self, name, state, location, interactions):
        self.name = name
        self.state = state
        self.location = location
        self.interactions = interactions
        self.infection_severity = 0.0
        self.death_threshold = 0.1
        
    def update_state(self):
        
        if self.state == "healthy":
            if self.is_infected():
                self.state = "infected"
            elif self.has_died():
                self.state = "dead"
        elif self.state == "infected":
            self.infection_severity += 0.1
            if self.has_turned():
                self.state = "zombie"
            elif self.has_died():
                self.state = "dead"
        elif self.state == "zombie":
            if self.has_died():
                self.state = "dead"
    
    def is_infected(self):
        # Check if the person has come into contact with a zombie or infected individual
        for person in self.interactions:
            if person.state == "zombie" or person.state == "infected":
                return True
        return False
    
    def has_turned(self):
        # Calculate the probability of turning based on the infection severity
        p_severity = self.infection_severity
        # Check if the person has turned based on the calculated probability
        if random.random() < p_severity:
            return True
        return False
    
    def has_died(self):
        if self.state == "zombie":
            # Calculate the probability of death based on the age of the zombie
            death_probability = 1 - (1 / (1 + math.exp(self.death_threshold)))
        else:
            # Calculate the probability of death based on the severity of the infection
            death_probability = 1 - (1 / (1 + math.exp(-(self.infection_severity - self.death_threshold))))
            
            # Check if the zombie has died based on the calculated probability
            if random.random() < death_probability:
                return True
            
        return False

class SchoolZombieApocalypse:
    def __init__(self, school_size, population_size):
        # Create a 2D grid representing the school with each cell can contain a Person object
        self.school = [[[] for _ in range(school_size)] for _ in range(school_size)]
        # Create a list of Person objects representing the school's population
        self.population = []
        for i in range(population_size):
            state = random.choice(STATES)
            location = (random.randint(0, school_size), random.randint(0, school_size))
            self.population.append(Person(i, state=state, location=location, interactions=[]))
            # Assign each person to a cell in the school based on their location
            self.school[self.population[i].location[0]][self.population[i].location[1]].append(self.population[i])
        self.num_healthy = sum(1 for person in self.population if person.state == "healthy")
        self.num_infected = sum(1 for person in self.population if person.state == "infected")
        self.num_zombie = sum(1 for person in self.population if person.state == "zombie")
        self.num_dead = sum(1 for person in self.population if person.state == "dead")
        self.population_size = self.num_healthy + self.num_infected + self.num_zombie + self.num_dead
        self.severity = 0
        self.probability_of_infection = 1 - (1 / (1 + math.exp(-self.severity)))
        self.probability_of_turning = 1 - (1 / (1 + math.exp(-self.severity)))
        self.probability_of_death = self.severity

    def run_simulation(self, num_time_steps):
        # run simulation for a specified number of time steps
        for time in range(num_time_steps):
            # update cellular automaton grid
            self.update_grid()
            self.update_states()
            # update population size based on current states
            self.update_population_size()
            self.severity = time / num_time_steps
            
    # update the states of each person in the population based on their interactions with other people      
    def update_grid(self):
        # Iterate through each cell in the school
        for i in range(len(self.school)):
            for j in range(len(self.school[i])):
                cell = self.school[i][j]
                # Get the states of the cell's neighbors
                neighbors = self.get_neighbors(self.school, i, j)
                num_healthy_neighbors = sum(1 for n in neighbors if n == "healthy")
                num_infected_neighbors = sum(1 for n in neighbors if n == "infected")
                num_zombie_neighbors = sum(1 for n in neighbors if n == "zombie")
                
                # Update the positions of the zombies
                for person in cell:
                    if person.state == "zombie":
                        new_location = random.choice(self.get_neighbors(self.school, person.location[0], person.location[1]))
                        if new_location != None:
                            self.school[person.location[0]][person.location[1]].remove(person)
                            self.school[new_location[0]][new_location[1]].append(person)
                            person.location = new_location
                
                # Update the positions of the survivors
                for person in cell:
                    if person.state == "alive":
                        # Calculate the distance to the nearest zombie for each survivor
                        nearest_zombie_distance = float("inf")
                        nearest_zombie = None
                        for zombie in self.get_neighbors(self.school, person.location[0], person.location[1]):
                            if zombie.state == "zombie":
                                distance = abs(person.location[0] - zombie.location[0]) + abs(person.location[1] - zombie.location[1])
                                if distance < nearest_zombie_distance:
                                    nearest_zombie_distance = distance
                                    nearest_zombie = zombie
                        # Move the survivor one unit away from the nearest zombie
                        self.school[person.location[0]][person.location[1]].remove(person)
                        if nearest_zombie != None and nearest_zombie_distance == 0:
                            person.location[0] += -1 if nearest_zombie.location[0] > person.location[0] else 1
                            person.location[1] += -1 if nearest_zombie.location[1] > person.location[1] else 1
                        else:
                            person.location = (person.location[0] + random.choice([-1, 0, 1]), person.location[1] + random.choice([-1, 0, 1]))
                        self.school[person.location[0]][person.location[1]].append(person)
                        
                # Update the state of the cell's person based on the states of its neighbors
                if cell == "healthy":
                    if num_zombie_neighbors > 0:
                        # Person becomes infected with a certain probability based on the severity of the outbreak
                        probability_of_infection = self.severity * self.probability_of_infection
                        if random.random() < probability_of_infection:
                            updated_cell = "infected"
                        else:
                            updated_cell = "healthy"
                    else:
                        updated_cell = "healthy"
                elif cell == "infected":
                    if num_healthy_neighbors + num_infected_neighbors > 0:
                        # Person infected may be killed with a certain probability based on the severity of the outbreak
                        probability_of_killed = self.severity * self.probability_of_death
                        if random.random() < probability_of_killed:
                            updated_cell = "dead"
                        else:
                            updated_cell = "infected"
                    # Person turns into a zombie with a certain probability based on the severity of the outbreak
                    probability_of_turning = self.severity * self.probability_of_turning
                    if random.random() < probability_of_turning:
                        updated_cell = "zombie"                    
                    else:
                        updated_cell = "infected"
                elif cell == "zombie":
                    if num_healthy_neighbors + num_infected_neighbors > 0:
                        # Zombie has a certain probability of death based on the severity of the outbreak
                        probability_of_killed = self.severity * self.probability_of_death
                        if random.random() < probability_of_killed:
                            updated_cell = "dead"
                        else:
                            updated_cell = "zombie"
                    else:
                        updated_cell = "zombie"
                else:
                    # Other states (e.g., "dead") do not change
                    updated_cell = cell
                    
                # Update the state of the cell's person
                if updated_cell != cell:
                    for person in self.school[i][j]:
                        person.state = updated_cell
                        if updated_cell == "dead":
                            person = None
                            self.population.pop(self.population.index(person))
                        else:
                            person.location = (i, j)

    def get_neighbors(self, grid, x, y):
        """
        Returns a list of the states of the 8 cells surrounding the given cell (x, y) in the grid.
        If a cell is outside the grid, it is treated as if it is in the "dead" state.
        """
        neighbors = []
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if i == x and j == y:
                    # skip the center cell
                    continue
                if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[i]):
                    # cell is outside the grid
                    neighbors.append("dead")
                else:
                    neighbors.append(grid[i][j])
        return neighbors
    
    # Determine the state of the cell based on the states of the people within it and the rules or logic governing the behavior of the cellular automaton
    def update_states(self):
        for person in self.population:
            person.update_state()
    
    def update_population_size(self):
        self.num_healthy = sum(1 for person in self.population if person.state == "healthy")
        self.num_infected = sum(1 for person in self.population if person.state == "infected")
        self.num_zombie = sum(1 for person in self.population if person.state == "zombie")
        self.num_dead = sum(1 for person in self.population if person.state == "dead")
        self.population_size = self.num_healthy + self.num_infected + self.num_zombie + self.num_dead
        self.probability_of_infection = 1 - (1 / (1 + math.exp(-self.severity)))
        self.probability_of_turning = 1 - (1 / (1 + math.exp(-self.severity)))
        self.probability_of_death = self.severity
        
    def observe_population(self):
        # Count the number of individuals in each state
        num_healthy = self.num_healthy
        num_infected = self.num_infected
        num_zombie = self.num_zombie
        num_dead = self.num_dead
        
        # Calculate the percentage of cells in each state
        healthy_percent = num_healthy / (num_healthy + num_infected + num_zombie)
        infected_percent = num_infected / (num_healthy + num_infected + num_zombie)
        zombie_percent = num_zombie / (num_healthy + num_infected + num_zombie)

        
        # Calculate the rate of infection, turning, and death
        infection_rate = num_infected / (num_healthy + num_infected)
        turning_rate = num_zombie / num_infected
        death_rate = num_dead / (num_infected + num_zombie)
        
        # Print the results
        print("Number of healthy individuals:", num_healthy)
        print("Number of infected individuals:", num_infected)
        print("Number of zombie individuals:", num_zombie)
        print("Number of dead individuals:", num_dead)
        print("Percentage of healthy cells:", healthy_percent)
        print("Percentage of infected cells:", infected_percent)
        print("Percentage of zombie cells:", zombie_percent)
        print("Infection rate:", infection_rate)
        print("Turning rate:", turning_rate)
        print("Death rate:", death_rate)

    def observe_school(self):
        # Print a visual representation of the school, with each cell represented by a character
        for row in self.school:
            for cell in row:
                if cell == "healthy":
                    print("H", end="")
                elif cell == "infected":
                    print("I", end="")
                elif cell == "zombie":
                    print("Z", end="")
                elif cell == "empty":
                    print("E", end="")
            print()
            
        # create a scatter plot of the population
        # x = [person.location[0] for person in self.population]
        # y = [person.location[1] for person in self.population]
        cell_states = [person.state for person in self.population]
        # plt.scatter(x, y, c=cell_states)
        plt.scatter(*zip(*self.population), c=cell_states)
        plt.show()
        
        # Analyze the results by observing the changes in the population over time
        cell_states = [person.state for person in self.population]
        counts = {state: cell_states.count(state) for state in STATES}
        
        # Add a bar chart to show the counts of each state in the population
        plt.bar(list(counts.keys()), list(counts.values()))
        
        # Show the plot
        plt.show()
        
# Create a SchoolZombieApocalypse object
school_sim = SchoolZombieApocalypse(school_size=10, population_size=10)

# Run the simulation for a given time period
school_sim.run_simulation(10)

# Observe the changes in the population and school over time
school_sim.observe_population()
school_sim.observe_school()

"""
# Define the rules or events that trigger transitions between states
# Determine whether the person has been infected based on their interactions with zombies or infected individuals
def is_infected(self, school, severity, tau, sigma, effectiveness):
    # Check if the person has come into contact with a zombie or infected individual
    for person in self.interactions:
        if person.state == "zombie" or person.state == "infected":
            # Calculate the probability of infection based on the duration of the interaction
            """
            # The probability of infection is being calculated based on the duration of the interaction between the person and another individual. The longer the interaction, the higher the probability of infection. The probability is being calculated using the formula 1 - e^(-duration/tau), where tau is a parameter representing the average time it takes for the infection to be transmitted. The exponent in the formula is negative because a longer duration means a higher probability of infection, and the negative exponent means that the probability decreases as the duration increases. The final probability is calculated by subtracting this value from 1, meaning that the probability increases as the duration increases.
            """
            probability = 1 - math.exp(-self.interaction_duration / tau)
            
            # Calculate the probability of infection based on the distance between the two individuals
            row1, col1 = self.location
            row2, col2 = person.location
            distance = math.sqrt((row1 - row2)**2 + (col1 - col2)**2)
            """
            # This line of code is updating the probability of infection based on the distance between the two individuals. The probability is being calculated using the formula 1 - e^(-distance/sigma), where sigma is a parameter representing the average distance at which the infection can be transmitted. The exponent in the formula is negative because a shorter distance means a higher probability of infection, and the negative exponent means that the probability decreases as the distance increases. The probability is then being updated by multiplying it by this value, meaning that the overall probability will decrease as the distance increases.
            """
            probability *= 1 - math.exp(-distance / sigma)
            
            # Multiply the probability by the effectiveness of any protective measures
            """
            # This line of code is updating the probability of infection based on the effectiveness of any protective measures that the person may be using. For example, if the person is wearing a mask or gloves, the probability of infection may be lower. The probability is being updated by multiplying it by the effectiveness value, which represents the degree to which the protective measures are effective at preventing infection. If the effectiveness value is 1, it means that the measures are completely effective and the probability will not change. If the effectiveness value is less than 1, it means that the measures are less effective and the probability will increase.
            """
            probability *= effectiveness
            
            # Multiply the probability by the overall severity of the outbreak
            """
            # This line of code is updating the probability of infection based on the overall severity of the zombie outbreak. The probability is being updated by multiplying it by the severity value, which represents the overall severity of the outbreak on a scale from 0 to max_severity. If the severity value is 0, it means that the outbreak is not severe and the probability will not change. If the severity value is greater than 0, it means that the outbreak is more severe and the probability will increase. The probability is also being divided by the max_severity value, which represents the maximum possible severity of the outbreak. This is being done to normalize the probability so that it is always between 0 and 1.
            """
            probability *= severity / school.max_severity
            
            # Return True if the probability is greater than a random number, False otherwise
            return random.random() < probability
    return False

# Determine whether the person has turned into a zombie based on the passage of time or the severity of the infection
def has_turned(self):
    # Calculate the probability of turning based on the infection severity
    """
    # calculates the probability of turning into a zombie based on the severity of the infection. The higher the severity, the higher the probability of turning.
    """
    p_severity = 1 - (1 / (1 + self.infection_severity))
    
    # Calculate the probability of turning based on the effectiveness of treatment
    """
    # calculates the probability of turning into a zombie based on the effectiveness of treatment. The higher the treatment effectiveness, the lower the probability of turning.
    """
    p_treatment = 1 - self.treatment_effectiveness
    
    # Calculate the overall probability of turning
    """
    # calculates the overall probability of turning into a zombie based on both the severity of the infection and the effectiveness of treatment.
    """
    p_turning = p_severity * p_treatment
    
    # Check if the person has turned based on the calculated probability
    if random.random() < p_turning:
        return True
    
    return False


# Determine whether the person has died based on the severity of the infection or other causes such as injuries or starvation
def has_died(self):
    if self.state == "zombie":
        # Calculate the probability of death based on the age of the zombie
        """
        # This code calculates the probability of death for a zombie based on the value of DEATH_THRESHOLD.
        """
        death_probability = 1 - (1 / (1 + math.exp(DEATH_THRESHOLD)))
        
        # Adjust the probability of death based on the availability of resources or external factors
        if self.has_resources:
            death_probability *= 0.5
        if self.exposed_to_sunlight:
            death_probability *= 0.75
        if self.injuries > 0:
            death_probability *= 1.25
        
        # Check if the zombie has died based on the calculated probability
        if random.random() < death_probability:
            return True
        
        return False
    else:
        # Calculate the probability of death based on the severity of the infection
        """
        # This code calculates the probability of death for a person who is not a zombie based on the severity of their infection and a value called DEATH_THRESHOLD. By subtracting DEATH_THRESHOLD from the person's infection severity, this code is able to take into account the fact that people with more severe infections are more likely to die.
        """
        death_probability = 1 - (1 / (1 + math.exp(-(self.infection_severity - DEATH_THRESHOLD))))
        
        # Adjust the probability of death based on the availability of medical treatment or resources
        if self.has_treatment:
            death_probability *= 0.5
        if self.has_resources:
            death_probability *= 0.75
        if self.injuries > 0:
            death_probability *= 1.25
        
        # Check if the person has died based on the calculated probability
        if random.random() < death_probability:
            return True
        
        return False
"""
"""
Define the rules of the simulation
Zombie infection - if a zombie and survivor are in neighbouring cell, the survivor will become infected
Survivor attack - if a zombie and survivor are in neighbouring cell, if a zombie dies, it is removed from the simulation
Zombie movement - each zombie moves one unit towards a random direction
Survivor movement - each survivor moves one unit away from the nearest zombie

a: individual: zombie and survivor, cell: position, grid: zombie_positions and survivor_positions, simulation: update_simulation()

Person.interactions

# Define the rules or events that trigger transitions between states in the population model
def determine_event():
    # Determine the probability of each event occurring
    infection_probability = 0.1
    turning_probability = 0.05
    death_probability = 0.01
    
    # Generate a random number between 0 and 1
    r = random.random()
    
    # Compare the random number to the probabilities of each event to determine which event should be triggered
    if r < infection_probability:
        return "infection"
    elif r < infection_probability + turning_probability:
        return "turning"
    elif r < infection_probability + turning_probability + death_probability:
        return "death"
    else:
        return None
"""

"""
Here are a few additional considerations that you may want to take into account when implementing the simulation:

Data collection and storage: You may want to consider how you will store and track data about the attributes of each individual, such as their age, gender, location, and state. This could involve creating a database or data structure to store this information.

Visualization: It may be helpful to visualize the simulation in some way, such as by creating a graphical user interface or using a visualization tool like Matplotlib. This can make it easier to understand the results of the simulation and identify trends or patterns.

Validation: It's important to validate the accuracy of the simulation by comparing the results to real-world data or known facts about how zombie outbreaks spread. This can help ensure that the simulation is a realistic and accurate representation of the scenario it is modeling.

Sensitivity analysis: It may be useful to perform sensitivity analysis to understand how the simulation results change as different parameters or assumptions are altered. For example, you could vary the rate of infection or the effectiveness of containment measures and see how these changes affect the outcome of the simulation.

Extension: You may want to consider extending the simulation to include additional factors or scenarios. For example, you could incorporate the behavior of external actors, such as emergency responders or military personnel, or model the spread of the zombie virus to other locations outside the school.
"""