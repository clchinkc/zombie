# define the possible states
LOCATION = "location"
ZOMBIE_NEARBY = "zombie_nearby"
RESOURCES = "resources"

# define the possible actions
MOVE = "move"
SEARCH = "search"
FIGHT = "fight"

# define the rewards
KILL_ZOMBIE = 100
DIE_BY_ZOMBIE = -1000
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.999

# initialize the Q-table with default values of 0
Q = {
    (LOCATION, ZOMBIE_NEARBY, RESOURCES): {
        MOVE: 0,
        SEARCH: 0,
        FIGHT: 0
    },
    ("safe_house", ZOMBIE_NEARBY, RESOURCES): {
        MOVE: 0,
        SEARCH: 0,
        FIGHT: 0
    }
}

# define the reinforcement learning algorithm
def q_learning(state, action, reward, next_state):
    # calculate the maximum expected reward for the next state
    max_reward = max(Q[next_state][a] for a in Q[next_state])
    
    # update the Q-value for the current state and action
    Q[state][action] = (1 - LEARNING_RATE) * Q[state][action] + LEARNING_RATE * (reward + DISCOUNT_RATE * max_reward)

# define the function for executing an action
def execute_action(state, action):
    if action == MOVE:
        # move to a new location and update the state
        next_state = ("safe_house", ZOMBIE_NEARBY, RESOURCES)
        return next_state, 0
    elif action == SEARCH:
        # search for a weapon and update the state and reward
        next_state = (LOCATION, ZOMBIE_NEARBY, RESOURCES + 1)
        if RESOURCES > 0:
            reward = 0
        else:
            reward = 0
        return next_state, reward
    elif action == FIGHT:
        # fight a nearby zombie and update the state and reward
        next_state = (LOCATION, ZOMBIE_NEARBY - 1, RESOURCES - 1)
        if ZOMBIE_NEARBY == 0:
           reward = KILL_ZOMBIE
        else:
            reward = 0
    return next_state, reward



# define the function for checking the end condition
def end_condition(state):
    # end the simulation if the person is killed by a zombie
    if int(state[1]) > 0:
        return True
    else:
        return False
    
# simulate the person's behavior
state = (LOCATION, ZOMBIE_NEARBY, RESOURCES)
while True:
    # select an action according to the learned policy
    action = max(Q[state], key=Q[state].get)
    
    # execute the action and observe the resulting state and reward
    next_state, reward = execute_action(state, action)
    
    # update the Q-table using the reinforcement learning algorithm
    q_learning(state, action, reward, next_state)
    
    # set the current state to the next state
    state = next_state
    
    # check if the simulation should end
    if end_condition(state):
        break
    
    # print the person's current action
    print(f"The person is {action}.")
