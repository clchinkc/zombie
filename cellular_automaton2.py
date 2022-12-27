
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class Agent:
    def __init__(self, location, health, strength):
        self.location = location
        self.health = health
        self.strength = strength

    def move(self, direction):
        self.location += direction

    def attack(self, agent):
        agent.health -= self.strength

class Human(Agent):
    def __init__(self, location, health, strength):
        super().__init__(location, health, strength)

    def move(self, direction):
        super().move(direction)

    def attack(self, agent):
        super().attack(agent)


class Zombie(Agent):
    def __init__(self, location, health, strength):
        super().__init__(location, health, strength)

    def move(self, direction):
        super().move(direction)

    def attack(self, agent):
        super().attack(agent)


class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(self.width)]
                     for _ in range(self.height)]
        self.init_agents()

    def set_cell(self, x, y, agent):
        self.grid[x][y] = agent
        
    def get_cell(self, x, y):
        return self.grid[x][y]

    def init_agents(self):
        # set the initial state of the cells
        for i in range(self.width):
            for j in range(self.height):
                if np.random.random() < 0.1:
                    # Place a human individual with probability 0.1
                    human = Human((i, j), 100, 10)
                    self.set_cell(i, j, human)
                elif np.random.random() < 0.1:
                    # Place a zombie individual with probability 0.1
                    zombie = Zombie((i, j), 100, 10)
                    self.set_cell(i, j, zombie)
                else:
                    continue
        """
        while True:
            # Place one more zombie individual 
            random_i = np.random.randint(0, self.width)
            random_j = np.random.randint(0, self.height)
            if self.get_cell(random_i, random_j) == None:
                self.set_cell(random_i, random_j, Zombie)
                break
        """

    def legal_move(self, agent, direction):
        if direction == (0, 0):
            return True
        if agent.location[0] + direction[0] < 0 or agent.location[0] + direction[0] > self.width - 1:
            return False
        if agent.location[1] + direction[1] < 0 or agent.location[1] + direction[1] > self.height - 1:
            return False
        if self.grid[agent.location[0] + direction[0]][agent.location[1] + direction[1]] != None:
            return False
        return True

    def move_agent(self, agent, direction):
        if self.legal_move(agent, direction):
            old_location = agent.location
            agent.move(direction)
            self.grid[agent.location[0]][agent.location[1]] = agent
            self.grid[old_location[0]][old_location[1]] = None
            return True
        else:
            return False

    def attack_agent(self, agent, target):
        agent.attack(target)
        if target.health <= 0:
            if isinstance(target, Human):
                zombie = Zombie(target.location, 100, 10)
                self.set_cell(target.location[0],target.location[1], zombie)
            else:
                self.set_cell(target.location[0],target.location[1], None)
            return True
        else:
            return False
        """
        self.infection_rate and even if health > 0, still have a chance to get infected
        zombie health == human health before infection
        """

    def get_neighbors(self, agent):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if agent.location[0] + i < 0 or agent.location[0] + i > self.width - 1:
                    continue
                if agent.location[1] + j < 0 or agent.location[1] + j > self.height - 1:
                    continue
                if self.grid[agent.location[0] + i][agent.location[1] + j] == None:
                    continue
                neighbors.append(
                    self.grid[agent.location[0] + i][agent.location[1] + j])
        return neighbors

    def choose_direction(self, agent):
        neighbors = self.get_neighbors(agent)
        if len(neighbors) == 0:
            while True:
                random_direction = (random.randint(-1, 1),
                                    random.randint(-1, 1))
                if self.legal_move(agent, random_direction):
                    return random_direction
        if isinstance(agent, Human):
            for neighbor in neighbors:
                if isinstance(neighbor, Zombie) and self.legal_move(agent, (agent.location[0] - neighbor.location[0], agent.location[1] - neighbor.location[1])):
                    return (agent.location[0] - neighbor.location[0], agent.location[1] - neighbor.location[1])
            human_locations = []
            for neighbor in neighbors:
                    if isinstance(neighbor, Human):
                        human_locations.append(neighbor.location)
            while True:
                human_distances = [np.linalg.norm(
                    agent.location - hp) for hp in human_locations]
                closest_human = human_locations[np.argmin(human_distances)]
                direction = closest_human - agent.location
                if self.legal_move(agent, direction):
                    return direction
                else:
                    human_locations.remove(closest_human)
                if len(human_locations) == 0:
                    break
        elif isinstance(agent, Zombie):
            for neighbor in neighbors:
                if isinstance(neighbor, Human):
                    if self.legal_move(agent, (neighbor.location[0] - agent.location[0], neighbor.location[1] - agent.location[1])):
                        return (neighbor.location[0] - agent.location[0], neighbor.location[1] - agent.location[1])
        else:
            random_direction = (random.randint(-1, 1), random.randint(-1, 1))
            return random_direction
        
        """
        cases of more than one zombie and human
        cases when both zombie and human are in neighbors
        """
        """
        use random walk algorithm to simulate movement based on probability adjusted by cell's infection status and location
        """
        """
        def simulate_movement(self):
            for i in range(self.width):
                for j in range(self.height):
                    individual = self.grid[i][j]
                    if individual is not None:
                        # use the A* algorithm to find the shortest path to the nearest exit
                        start = (i, j)
                        # the four corners of the grid
                        exits = [(0, 0), (0, self.width-1),
                                (self.height-1, 0), (self.height-1, self.width-1)]
                        distances, previous = self.a_star(start, exits)
                        # use the first exit as the destination
                        path = self.reconstruct_path(previous, start, exits[0])

                        # move to the next cell in the shortest path to the nearest exit
                        if len(path) > 1:  # check if there is a valid path to the nearest exit
                            next_x, next_y = path[1]
                            # update the individual's location
                            individual.location = (next_x, next_y)
                            # remove the individual from their current location
                            self.grid[i][j] = None
                            # add the individual to their new location
                            self.grid[next_x][next_y] = individual

        def a_star(self, start, goals):
            # implement the A* algorithm to find the shortest path from the start to one of the goals
            # returns the distances and previous nodes for each node in the grid
            pass

        def reconstruct_path(self, previous, start, goal):
            # implement the algorithm to reconstruct the path from the previous nodes
            # returns the shortest path from the start to the goal
            pass
        """
        
    def attack_neighbors(self, agent):
        neighbors = self.get_neighbors(agent)
        if isinstance(agent, Human):
            for neighbor in neighbors:
                if isinstance(neighbor, Zombie):
                    self.attack_agent(agent, neighbor)
        elif isinstance(agent, Zombie):
            for neighbor in neighbors:
                if isinstance(neighbor, Human):
                    self.attack_agent(agent, neighbor)

    def choose_action(self, agent):
        neighbors = self.get_neighbors(agent)
        if isinstance(agent, Human):
            for neighbor in neighbors:
                if isinstance(neighbor, Zombie):
                    return "attack"
            return "move"
        elif isinstance(agent, Zombie):
            for neighbor in neighbors:
                if isinstance(neighbor, Human):
                    return "attack"
            return "move"

    def update(self):
        all_agents = []
        for i in range(self.width):
            for j in range(self.height):
                agent = self.grid[i][j]
                if agent == None:
                    continue
                all_agents.append(agent)
        for agent in all_agents:    
            action = self.choose_action(agent)
            if action == "move":
                direction = self.choose_direction(agent)
                self.move_agent(agent, direction)
            elif action == "attack":
                self.attack_neighbors(agent)
            else:
                continue
        """
        # check if any surrounding cells are infected and update the current cell's state accordingly
        for cell in surrounding_cells:
            if self.grid[cell[0]][cell[1]].state == "infected":
                self.state = "infected"
                break
        """

    def plot_grid(self):
        grid_plot = np.zeros((self.width, self.height))
        color_plot = np.zeros((self.width, self.height))
        for i in range(self.width):
            for j in range(self.width):
                if self.grid[i][j] == None:
                    grid_plot[i][j] = "-"
                    color_plot[i][j] = "gray"                  
                elif isinstance(self.grid[i][j], Human):
                    grid_plot[i][j] = "H"
                    color_plot[i][j] = "green"
                elif isinstance(self.grid[i][j], Zombie):
                    grid_plot[i][j] = "Z"
                    color_plot[i][j] = "red"
        plt.scatter(range(self.width), range(self.height), c=color_plot)
        plt.figure()
        plt.show()
        plt.pause(0.1)
            
def run_simulation(width, height, num_steps):
    grid = Grid(width, height)
    for i in range(num_steps):
        grid.update()
    return grid

"""
def run_simulation(self, iterations):
    for _ in range(iterations):
        # update the locations of the agents and the states of the cells
        for agent in self.agents:
            agent.move()
            agent.fight()
        for row in self.grid:
            for cell in row:
                cell.update_state()
    """
    
grid = run_simulation(100, 100, 100)
grid.plot_grid()
