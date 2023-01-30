"""
The simulation models the model the behaviour and attributes of individual survivors and zombies in a population of individuals in zombie apocalypse at school. At each time step of the simulation, the following steps are performed:

Update the states of the individuals: For each individual (separate for humans and zombies) in the population, determine and update the appropriate transition to the next state based on the individual's attributes and current state.

Update the states of the cells on the grid: For each cell on the grid, determine the movement and state of the cell based on the individuals (humans and zombies) in the neighbouring cells.

Update any relevant statistics or outputs: Keep track of any relevant statistics or outputs of the simulation.

Repeat these steps for the desired time period of the simulation.

This simulation combines a state machine model (which models the transitions between the different states that an individual can be in, ie healthy, infected, zombie, and dead) with a cellular automaton model (which represents the locations of the individuals and the state of each location on a grid). It allows for tracking the behaviour and states of the individuals and the movement and interaction through the population over time.
"""

"""
To implement this simulation, we will need to define the following in Python:

Define the relevant variables:
num_zombies: This will be an integer variable that represents the number of zombies at the school.
num_survivors: This will be an integer variable that represents the number of survivors at the school.
zombie_locations: This will be a list of tuples, where each tuple represents the (x, y) coordinates of a zombie in the school.
survivor_locations: This will be a list of tuples, where each tuple represents the (x, y) coordinates of a survivor in the school.
infection_rate: This will be a float variable that represents the probability that a survivor will become infected after being bitten by a zombie.
turning_rate: This will be a float variable that represents the probability that an infected survivor will turn into a zombie.
death_rate: This will be a float variable that represents the probability that a zombie will die after being attacked by a survivor.

Define the states:
"alive" - representing individuals who are alive and not infected with the zombie virus
"infected" - representing individuals who are infected with the zombie virus but not yet turned into zombies
"zombie" - representing individuals who have turned into zombies
"dead" - representing individuals who have died, either from the zombie virus or other causes

Define the transitions between states:
"infection" - representing the process by which humans become infected
"turn" - representing the process by which infected humans turn into zombies
"death" - representing the process by which humans and zombies die

Define the rules or events that trigger transitions between states that govern the behaviour and interactions between humans and zombies:
"infecting" is triggered by an individual coming into contact with a zombie or an infected individual (between cells)
"turning" is triggered by the passage of time or the severity of the infection (within the cell)
"dying" is triggered by the severity of the infection or other causes such as injuries or starvation (between cells or within the cell)
"move" is triggered when other rules are not triggered and changes the location of humans and zombies

Define classes:
Individual: This class represents an individual in the population. It should have attributes for the individual's state (e.g. healthy, infected, zombie, dead) and any other relevant attributes (e.g. health, strength). It should also have a method get_transition() that implement the transition between state and change individual's attributes according to current state and other rules implemented in cell and grid class. It also checks if the individual is infected or dies.
Cell: This class represents a cell on the grid of the cellular automaton. It should have an attribute for the possibility of transition between states (e.g. infection, turn, death) and a method update_state() that updates the state of the individual based on the states of the cells and the individual's attributes. For example, if the individual is infected, the update_state() method should return "turning" with probability turning_rate.
Grid: This class represents the grid of the cellular automaton. It should have an attribute for the size of the grid. It should have a method create_grid() that initializes the grid with a set of cells and assigns each cell a location. It should also have a method update_cell_states() that updates the states of the cells (e.g. infection rate, turning rate, dying rate, move rate) on the grid based on the time step as well as states of the individuals in neighbouring cells. For example, if a cell has an individual that is a neighbour of zombie, the update_cell_states() method return "infecting" with probability infecting_rate. It should also have a move() method that moves the individuals in the grid according to whether the move is legal.
Simulation: This is the main class for the simulation. It should have a method run() that runs the simulation for a given time period. At each time step, it should call the get_transition() method of the Individual class to update the states of the individuals in the population, use update_state() to update the state of the cells, and use the update_cell_states() method of the Grid class to update the states of the cells across the grid. It should also keep track of any relevant statistics and outputs of the simulation, such as the current time step, the change of population, the number of individuals in each state, the rate of infection, turn, and death. It should also have a info() method that prints info of each individual in the population and plot() method that plots the grid and color-code the cells inside, at the end of the simulation.


update_cell_states() updates the cells at each iteration. The function first checks if any zombies are in the same position as any survivors, and if so, there is a chance that the survivor will become infected based on the zombie_infection_rate. If the survivor becomes infected, the number of zombies increases by 1 and the number of survivors decreases by 1. The survivor's position is then added to the list of zombie positions, and the survivor is removed from the list of survivor positions. The function then let humans attack zombies and checks if any of the zombies are killed by survivors. If so, the zombie is removed from the list of zombie positions and the number of zombies decreases by 1. Next, the function moves each zombie by one unit in a random direction. Then, it calculates the distance between each survivor and the nearest zombie, and moves each survivor one step away from the closest zombie.

"""

"""
Testing and refining a simulation is an important step in the development process to ensure that it is accurately representing the real-world system or process that it is meant to model. There are several steps you can take to test and refine your simulation:

Run the simulation multiple times with different input parameters to see how it behaves under different conditions.

Compare the results of the simulation to real-world data or observations to see how well the simulation matches reality.

Identify any discrepancies between the simulation and real-world data, and adjust the rules or parameters of the simulation to improve its accuracy.

Consider adding additional features or complexity to the simulation to make it more realistic or engaging. This could include adding more variables or interactions between elements in the simulation, or incorporating additional data sources to make the simulation more comprehensive.

Continuously test and refine the simulation as necessary to ensure that it is accurate and representative of the real-world system or process it is meant to model.
"""

"""
Additional details:

You could also define other events as the rule that trigger transitions between states, such as coming into contact with a zombie or an infected individual, the passage of time, the severity of the infection, or other causes such as injuries or starvation.

It is also possible to further refine and enrich the simulation by incorporating additional factors or models. For example, you could include a resource model to represent the availability of food, water, medicine, and other resources, and incorporate rules or logic that govern how these resources are used and replenished. You could also include a model for external actors, such as government agencies or other survivors, and incorporate their behavior and interactions with the school population. By incorporating these additional models and factors, you can create a more realistic and comprehensive simulation of the zombie apocalypse at a school.

https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/
"""

