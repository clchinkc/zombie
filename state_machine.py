"""
It is possible to use a state machine approach to simulate a person's activities during a zombie apocalypse at school. 
A state machine is a mathematical model that is used to describe the behavior of a system 
by identifying the possible states it can be in and the transitions between those states. 
In the case of a person trying to survive a zombie apocalypse at school, 
the states could include things like hiding, searching for supplies, fighting zombies, and so on. 
The transitions between these states would depend on the specific circumstances and events that the person experiences. 
For example, if the person is currently hiding and they hear a noise nearby, 
they might transition to a state of searching for the source of the noise to determine if it is a threat. 
Similarly, if the person is currently fighting zombies and they run out of weapons, 
they might transition to a state of searching for supplies to find more weapons. 
By modeling the person's activities in this way, it would be possible to simulate their behavior 
and predict how they might act in different situations.

Here is a possible python simulation of a person's behavior during a zombie apocalypse at school:
"""

import random
import numpy as np


# define the possible states
# determined by actions and events
HIDING = "hiding"
SEARCHING = "searching"
FIGHTING = "fighting"
states = [HIDING, SEARCHING, FIGHTING]

# define the possible actions
# determined by states and tactics
HIDE = "hide"
SEARCH = "search"
FIGHT = "fight"
actions = [HIDE, SEARCH, FIGHT]

# define the possible events
# determined by states, actions, and possibilities
NOISE = "noise"
ZOMBIE = "zombie"
WEAPON = "weapon"
events = [NOISE, ZOMBIE, WEAPON]

# define the rewards
KILL_ZOMBIE = 10
DIE_BY_ZOMBIE = -100
SURVIVE = 1
FIND_RESOURCES = 5
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.999

# define the number of states and actions
num_states = len(states)
num_actions = len(actions)
num_events = len(events)

# define the school class to store states and produce events
class School(object):
    def __init__(self):
        self.state = HIDING
        # define the matrix of states for each state and event
        self.state_transitions = {
            HIDING: {
                NOISE: SEARCHING,
                ZOMBIE: FIGHTING,
                WEAPON: HIDING,
            },
            SEARCHING: {
                NOISE: SEARCHING,
                ZOMBIE: FIGHTING,
                WEAPON: HIDING,
            },
            FIGHTING: {
                NOISE: HIDING,
                ZOMBIE: FIGHTING,
                WEAPON: SEARCHING,
            },
        }
        self.event = NOISE
        # define the matrix of events for each state and action
        self.event_probabilities = [
        # NOISE, ZOMBIE, WEAPON
        [[0.11, 0.12, 0.77],    # HIDE, HIDING
        [0.13, 0.14, 0.73],    # SEARCH, HIDING
        [0.15, 0.16, 0.69]],   # FIGHT, HIDING
        [[0.17, 0.18, 0.65],    # HIDE, SEARCHING
        [0.19, 0.20, 0.61],    # SEARCH, SEARCHING
        [0.21, 0.22, 0.57]],   # FIGHT, SEARCHING
        [[0.23, 0.24, 0.53],    # HIDE, FIGHTING
        [0.25, 0.26, 0.49],    # SEARCH, FIGHTING
        [0.27, 0.28, 0.45]]   # FIGHT, FIGHTING
        ]

    def update(self, action):
        # select the next event
        self.event = self.produce_event(action)
        # select the next state
        self.state = self.update_state(self.state, self.event)
        
    def update_state(self, state, event):
        # select the next state based on the current state and event
        return self.state_transitions[state][event]
        
    def produce_event(self, action):
        # determine the event based on the state, action, and the event probabilities
        event_probabilities = self.event_probabilities[states.index(self.state)][actions.index(action)]
        return np.random.choice(events, p=event_probabilities)

# define the person class to hold the state machine and determine the action
class Person(object):
    def __init__(self):
        self.state_machine = School()
        self.action = HIDE
        # define the matrix of actions for each state
        self.action_probabilities = [
        # Hide, Fight, Run
        [0.31, 0.32, 0.37],    # HIDING
        [0.33, 0.34, 0.33],    # SEARCHING
        [0.35, 0.36, 0.29]     # FIGHTING
        ]

    def update(self):
        # update state and event
        self.state_machine.update(self.action)
        # select the next action
        self.action = self.select_action(self.state_machine.state)
        
    def select_action(self, state, randomness=1):
        # determine the action based on the state and the action probabilities
        state_index = states.index(state)
        action_probabilities = self.action_probabilities[state_index]
        if randomness:
            return np.random.choice(actions, p=action_probabilities)
        else:
            return actions[np.argmax(action_probabilities)]
        
# define the main function to run the simulation
def simulation():
    # define the number of people and the number of time steps
    num_people = 5
    num_steps = 100
    
    # create the people
    people = [Person() for i in range(num_people)]
    
    # run the simulation
    for step in range(num_steps):
        # update the people
        for person in people:
            person.update()
        
        # print the current state of the people
        print("step: ", step+1)
        print("state: ", [person.state_machine.state for person in people])
        print("action: ", [person.action for person in people])
        print("probabilities: ", [person.action_probabilities[states.index(person.state_machine.state)][actions.index(person.action)] for person in people])
        print("events: ", [person.state_machine.event for person in people])
        print("probabilities: ", [person.state_machine.event_probabilities[states.index(person.state_machine.state)][actions.index(person.action)][events.index(person.state_machine.event)] for person in people])
        print()
# run the simulation
simulation()



"""
# use neural network in the action selection process
def select_action(person, event):
    # Create a list of the current state and event
    state_event = [person.state.is_hiding, person.state.is_fighting, person.state.is_running, event == "Safe", event == "Overrun", event == "Noise"]

    # Create a list of the probabilities of selecting each action
    probabilities = nn.predict(state_event)

    # Select the next action based on the probabilities
    action = np.random.choice(actions, p=probabilities)

    # Return the next action
    return action

# training of the neural network using the simulation function
def train_nn(nn, steps, epochs):
    # Iterate over the number of epochs
    for _ in range(epochs):
        # Generate a person
        person = Person()

        # Simulate the person's behavior
        actions_taken = simulate(person, steps)

        # Create a list of the current states and events
        state_events = []
        for i in range(len(actions_taken)):
            state_events.append([person.state.is_hiding, person.state.is_fighting, person.state.is_running, actions_taken[i] == "Safe", actions_taken[i] == "Overrun", actions_taken[i] == "Noise"])

        # Create a list of the next states and events
        next_state_events = []
        for i in range(len(actions_taken)):
            next_state_events.append([person.state.is_hiding, person.state.is_fighting, person.state.is_running, actions_taken[i] == "Safe", actions_taken[i] == "Overrun", actions_taken[i] == "Noise"])

        # Train the neural network
        nn.train(state_events, next_state_events)

    return True
"""

"""
One potential advanced way to implement this would be to use a reinforcement learning (RL) algorithm 
to train the person's behavior in the zombie apocalypse. In this approach, 
the person's actions would be determined by a policy learned from experience rather than hard-coded rules. 
This could allow for more flexible and adaptable behavior in response to changing environmental conditions.

Another potential advanced implementation could be to use a multi-agent approach, 
where multiple people are simulated simultaneously and must interact with each other and the environment. 
This could add more complexity and realism to the simulation, 
as well as allowing for more sophisticated behaviors such as cooperation or competition between the people.

Additionally, the simulation could be extended to include more detailed information about the environment, 
such as the location and movement of zombies, the availability of resources, and the conditions of the buildings and surroundings. 
This could allow for more accurate and realistic simulations of the zombie apocalypse.
"""

"""
In a real-world scenario, it would be important to consider many additional factors and details
when simulating a person's behavior during a zombie apocalypse at school.
For example, the simulation could include information about the person's physical abilities,
the layout of the school and surrounding area, the availability of supplies and weapons,
and the movements and actions of other survivors and zombies.
Additionally, the simulation could include more complex behaviors and decision-making processes
to make the person's actions more realistic and unpredictable.
Furthermore, the simulation could be extended to include multiple people
and simulate their interactions and group dynamics.
Overall, a more detailed and realistic simulation would require a significant amount of planning
and development to accurately represent the complex and dynamic situation of a zombie apocalypse at school.

# a health, speed, and strength attribute for Person

# 2D map

# store information about the layout of the school and surrounding area, the availability of supplies and weapons, and the movements and actions of other survivors and zombies.

class Map:
    def __init__(self, size, supplies, weapons, survivors, zombies):
        self.layout = None
        self.supplies = supplies
        self.weapons = weapons
        self.survivors = survivors
        self.zombies = zombies

    def set_layout(self, layout):
        self.layout = np.random.choice(["Hallway", "Classroom", "Gym", "Cafeteria", "Library", "Auditorium",
                                       "Office", "Bathroom", "Locker Room", "Playground", "Parking Lot", "Field"], size=self.size)

    def get_status(self):
        return "There are " + str(self.supplies) + " supplies, " + str(self.weapons) + " weapons, " + str(self.survivors) + " survivors, and " + str(self.zombies) + " zombies."

"""