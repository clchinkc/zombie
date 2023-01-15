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

"""
Use a reinforcement learning (RL) algorithm to train the person's behavior in the zombie apocalypse.

To implement the reinforcement learning approach, we would first need to define a set of states, actions, 
and rewards that the person can experience in the zombie apocalypse. 
For example, the states could include the person's current location, the number of zombies nearby, 
and the availability of weapons or other resources. 
The actions could include moving to a new location, attacking a zombie, or hiding in a safe place. 
The rewards could reflect the person's survival, such as a positive reward for killing zombies 
and a negative reward for being killed by zombies.

Next, we would need to define a reinforcement learning algorithm, such as Q-learning, 
to learn a policy for selecting actions based on the current state and the observed rewards. 
This could be done by training the algorithm on a set of simulated scenarios, 
where the person's actions and the resulting rewards are recorded and used to update the policy.

Once the policy has been trained, we can use it to determine the person's actions in the simulation. 
This would involve observing the person's current state and selecting the action that is predicted 
to maximize the expected rewards according to the learned policy.

In this the code, we simulate the person's behavior for a fixed number of steps (1000 in this case) 
and choose a random event at each step. 
This allows the agent to learn from a variety of different scenarios and improve its policy for selecting actions. 
At the end of the simulation, we print the final q_values table to see how the agent has learned to select actions in each state.
"""

import random
import numpy as np
from functools import partial, update_wrapper

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

# define the school class to store states and produce events
class School(object):
    def __init__(self):
        self.state = np.random.choice(states)
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
        self.produce_event_globals = partial(self.produce_event, SURVIVE=1, FIND_RESOURCES=5, KILL_ZOMBIE=10, DIE_BY_ZOMBIE=-100)
        update_wrapper(self.produce_event_globals, self.produce_event)
        
        # may use probability to define the rules or events that trigger transitions between states
        
    def update_state(self, state, event):
        # select the next state based on the current state and event
        return self.state_transitions[state][event]
        
    def produce_event(self, state, action, SURVIVE, FIND_RESOURCES, KILL_ZOMBIE, DIE_BY_ZOMBIE):
        if state == HIDING:
            if action == HIDE:
                event = np.random.choice(events)
                reward = SURVIVE
            elif action == SEARCH:
                if np.random.random() < 0.5:
                    event = NOISE
                    reward = SURVIVE
                else:
                    event = ZOMBIE
                    reward = SURVIVE
            elif action == FIGHT:
                event = ZOMBIE
                reward = SURVIVE
            else:
                raise ValueError("Invalid action")
        elif state == SEARCHING:
            if action == HIDE:
                event = np.random.choice(events)
                reward = SURVIVE
            elif action == SEARCH:
                if np.random.random() < 0.5:
                    event = WEAPON
                    reward = FIND_RESOURCES
                else:
                    event = NOISE
                    reward = SURVIVE
            elif action == FIGHT:
                event = np.random.choice(events)
                reward = SURVIVE
            else:
                raise ValueError("Invalid action")
        elif state == FIGHTING:
            if action == HIDE:
                if np.random.random() < 0.5:
                    event = NOISE
                    reward = SURVIVE
                else:
                    event = ZOMBIE
                    reward = DIE_BY_ZOMBIE
            elif action == SEARCH:
                if np.random.random() < 0.3:
                    event = WEAPON
                    reward = SURVIVE
                elif np.random.random() < 0.5:
                    event = ZOMBIE
                    reward = SURVIVE
                else:
                    event = ZOMBIE
                    reward = DIE_BY_ZOMBIE
            elif action == FIGHT:
                if np.random.random() < 0.9:
                    if np.random.random() < 0.5:
                        event = WEAPON
                    else:
                        event = ZOMBIE
                    reward = KILL_ZOMBIE
                else:
                    event = ZOMBIE
                    reward = DIE_BY_ZOMBIE
            else:
                raise ValueError("Invalid action")
        else:
            raise ValueError("Invalid state")
        return event, reward

    """
    # execute action function with match case statement
    def execute_action(state, action):
        match state, action:
            case HIDING, HIDE:
                event = np.random.choice(events)
                reward = SURVIVE
            case HIDING, SEARCH:
                if np.random.random() < 0.5:
                    event = NOISE
                    reward = SURVIVE
                else:
                    event = ZOMBIE
                    reward = SURVIVE
            case HIDING, FIGHT:
                event = ZOMBIE
                reward = SURVIVE
            case SEARCHING, HIDE:
                event = np.random.choice(events)
                reward = SURVIVE
            case SEARCHING, SEARCH:
                if np.random.random() < 0.5:
                    event = WEAPON
                    reward = FIND_RESOURCES
                else:
                    event = NOISE
                    reward = SURVIVE
            case SEARCHING, FIGHT:
                event = np.random.choice(events)
                reward = SURVIVE
            case FIGHTING, HIDE:
                if np.random.random() < 0.5:
                    event = NOISE
                    reward = SURVIVE
                else:
                    event = ZOMBIE
                    reward = DIE_BY_ZOMBIE
            case FIGHTING, SEARCH:
                if np.random.random() < 0.3:
                    event = WEAPON
                    reward = SURVIVE
                elif np.random.random() < 0.5:
                    event = ZOMBIE
                    reward = SURVIVE
                else:
                    event = ZOMBIE
                    reward = DIE_BY_ZOMBIE
            case FIGHTING, FIGHT:
                if np.random.random() < 0.9:
                    if np.random.random() < 0.5:
                        event = WEAPON
                    else:
                        event = ZOMBIE
                    reward = KILL_ZOMBIE
                else:
                    event = ZOMBIE
                    reward = DIE_BY_ZOMBIE
            case _, _:
                raise ValueError("Invalid state or action")
    """
    """
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
    def produce_event(self, state, action):
        # determine the event based on the state, action, and the event probabilities
        event_probabilities = self.event_probabilities[states.index(self.state)][actions.index(action)]
        return np.random.choice(events, p=event_probabilities)
    """

# define the person class to hold the state machine and determine the action
class Person(object):
    def __init__(self):
        self.state_machine = School()
        self.action = np.random.choice(actions)
        self.reward = 0
        # define the matrix of actions for each state
        self.Q = np.zeros((len(states), len(actions)))
        self.update_q_values_globals = partial(self.update_q_values, LEARNING_RATE=0.001, DISCOUNT_RATE=0.999)
        update_wrapper(self.update_q_values_globals, self.update_q_values)

    def update(self):
        # select the next action
        self.action = self.select_action(self.state_machine.state, 0)
        # select the next event
        self.state_machine.event, reward = self.state_machine.produce_event_globals(self.state_machine.state, self.action)
        # select the next state
        self.state_machine.state = self.state_machine.update_state(self.state_machine.state, self.state_machine.event)
        return reward
    
    def end_condition(self, reward, die_by_zombie_value):
        # determine if the person has died based on reward
        return reward == die_by_zombie_value
    
    def select_action(self, state, epsilon=0):
        # choose an action using the learned epsilon-greedy policy
        state_index = states.index(state)
        if random.random() < epsilon:
            # choose a random action
            action_probabilities = self.softmax(self.Q[state_index])
            action = np.random.choice(actions, p=action_probabilities)
        else:
            # choose the action with the highest Q-value
            action = actions[np.argmax(self.Q[state_index, :])]
        return action
        
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)
        
    def update_q_values(self, state, action, reward, next_state, LEARNING_RATE, DISCOUNT_RATE):
        # calculate the maximum expected reward for the next state
        state_index = states.index(state)
        action_index = actions.index(action)
        next_state_index = states.index(next_state)
        next_max_reward = max(self.Q[next_state_index][:])
        # update the Q-value for the current state and action
        self.Q[state_index][action_index] = (1 - LEARNING_RATE) * self.Q[state_index][action_index] + \
            LEARNING_RATE * (reward + DISCOUNT_RATE * next_max_reward)

    def print_q_table(self):
        # print the Q-table
        print("Q-table:")
        for i in range(len(states)):
            print(f"State: {states[i]}")
            for j in range(len(actions)):
                print(f"Action: {actions[j]} - Q-value: {self.Q[i][j]}")
            print()

    # define the training function to train the person
    def train(self, num_episodes, epsilon, survive, find_resources, kill_zombie, die_by_zombie, learning_rate, discount_rate):
        # Loop through the episodes
        for episode in range(num_episodes):
            # print the current episode
            print(f"Episode: {episode+1}/{num_episodes}")
            
            # reset the state machine
            self.state_machine = School()
            # reset the action
            self.action = HIDE
            # Loop through the time steps
            while True:
                # select the next action
                self.action = self.select_action(self.state_machine.state, epsilon)
                # determine the event and reward
                self.state_machine.produce_event_globals = partial(self.state_machine.produce_event, SURVIVE=survive, FIND_RESOURCES=find_resources, KILL_ZOMBIE=kill_zombie, DIE_BY_ZOMBIE=die_by_zombie)
                self.event, self.reward = self.state_machine.produce_event_globals(self.state_machine.state, self.action)
                # determine the next state
                next_state = self.state_machine.update_state(self.state_machine.state, self.event)
                
                print(f"episode: {episode+1}, current state: {self.state_machine.state}, action: {self.action}, event: {self.state_machine.event}, reward: {self.reward}, next_state: {next_state}")
                
                # update the Q-values
                self.update_q_values_globals = partial(self.update_q_values, LEARNING_RATE=learning_rate, DISCOUNT_RATE=discount_rate)
                self.update_q_values_globals(self.state_machine.state, self.action, self.reward, next_state)
                
                # set the current state to the next state
                self.state_machine.state = next_state
                
                # determine if the episode is done
                if self.end_condition(self.reward, die_by_zombie):
                    break
                
        self.print_q_table()

# define the main function to run the simulation
def simulation():
    # define the number of people and the number of time steps
    num_people = 1
    num_steps = 1000
    
    # create the people
    people = [Person() for _ in range(num_people)]
    
    for person in people:
        # train the person
        person.train(num_episodes=100, epsilon=0.25, survive=1, find_resources=5, kill_zombie=10, die_by_zombie=-100, learning_rate=0.001, discount_rate=0.999)
    
    for step in range(num_steps):
        # run the simulation
        for person in people:
            # simulate the person
            person.update()
            # check end conditions
            if person.end_condition(person.reward, -100):
                break
        
        # print the current state of the people
        print("step: ", step+1)
        print("state: ", [person.state_machine.state for person in people])
        print("action: ", [person.action for person in people])
        print("probabilities: ", [person.softmax(person.Q[states.index(person.state_machine.state)])[actions.index(person.action)] for person in people])
        print("events: ", [person.state_machine.event for person in people])
        # print("probabilities: ", [person.state_machine.event_probabilities[states.index(person.state_machine.state)][actions.index(person.action)][events.index(person.state_machine.event)] for person in people])
        print()
        

        
# run the simulation
simulation()

"""
The code is implementing a reinforcement learning algorithm to teach a person to survive in a zombie apocalypse. It is using a Q-learning algorithm to determine the best actions to take in different states (such as "safe_house", "fighting", or "resources"). The person can either stay, move, or fight in each state. The rewards for these actions are defined as KILL_ZOMBIE, DIE_BY_ZOMBIE, SURVIVE, and FIND_RESOURCES. The Q-table is initialized with all zeros and is updated using the Q-learning formula, which takes into account the current reward and the expected maximum reward for the next state. The person's actions are selected using an epsilon-greedy policy, which sometimes chooses a random action and other times chooses the action with the highest Q-value. The training function loops through a specified number of episodes and performs the Q-learning update and action selection for each step until the end condition is reached (the person is killed by a zombie).
"""

"""
may try different values for learning rate, discount rate, and epsilon
"""

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