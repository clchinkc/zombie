import math
import random
    
class Agent:
    """Represents a game world agent with a position and health.
    
    Attributes:
        position (tuple): The position of the agent on the map.
        health (int): The health of the agent.
    """
    def __init__(self, position, health=100):
        self.position = position
        self.health = health
        
    def distance_to_agent(self, other_agent):
        """Calculate the distance between this agent and another agent.
    
        Args:
            other_agent (Agent): The other agent to calculate the distance to.
    
        Returns:
            float: The distance between the two agents.
        """
        x1, y1 = self.position
        x2, y2 = other_agent.position
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

class AgentManager:
    """Manages agents.

    Attributes:
        agents (list): A list of agents.
    """
    def __init__(self):
        self.agents = []
    
    def add_agent(self, agent):
        """Add an agent to the game world.
    
        Args:
            agent (Agent): The agent to add.
        """
        self.agents.append(agent)
    
    def remove_agent(self, agent):
        """Remove an agent from the game world.
    
        Args:
            agent (Agent): The agent to remove.
        """
        self.agents.remove(agent)
    
    def get_agents_in_range(self, agent, range):
        """Get a list of agents within a certain range of a given agent.
    
        Args:
            agent (Agent): The agent to check the range from.
            range (float): The range to check for agents within.
        
        Returns:
            list: A list of agents within the specified range.
        """
        in_range = []
        for other_agent in self.agents:
            if agent.distance_to_agent(other_agent) <= range:
                in_range.append(other_agent)
        return in_range

class Player(Agent):
    """Represents a player in the game world. Inherits from Agent.
    
    Attributes:
        inventory (list): A list of items in the player's inventory.
    """
    def __init__(self, position=(-1, -1), health=100):
        super().__init__(position, health)
        self.inventory = []
        
    def interaction_with_npc(self, npcs):
        """Interact with an NPC within a certain range.
        """
        for npc in npcs:
            if npc.interacting_with is None:
                npc.interacting_with = self
                self.perform_interaction(npc)
                break
            
    def perform_interaction(self, npc):
        """Randomly select one of three actions: trade with the NPC, fight with the NPC, or do nothing.
        
        Args:
            npc (NPC): The NPC to interact with.
        """
        num = random.random()
        if num < 0.3:
            self.trade_with_npc(npc)
        elif num < 0.6:
            self.fight_with_npc(npc)
        else:
            npc.interacting_with = None
    
    def trade_with_npc(self, npc):
        """Trade with an NPC.
        
        Args:
            npc (NPC): The NPC to trade with.
        """
        if len(self.inventory) > 0:
            if len(npc.inventory) > 0:
                self_item = random.choice(self.inventory)
                self.inventory.remove(self_item)
        
                npc_item = random.choice(npc.inventory)
                npc.inventory.remove(npc_item)
                
                self.inventory.append(npc_item)
                npc.inventory.append(self_item)
    
    def fight_with_npc(self, npc):
        """Fight with an NPC.
    
        Args:
            npc (NPC): The NPC to fight with.
        """
        while self.health > 0 and npc.health > 0:
            self.health -= random.randint(10, 20)
            npc.health -= random.randint(10, 20)
    
        if self.health <= 0:
            print("You lost the fight!")
        else:
            print("You won the fight!")


class NPC(Agent):
    """Represents an NPC in the game world. Inherits from Agent.
    
    Attributes:
        inventory (list): A list of items in the NPC's inventory.
        interacting_with (Player): The player that the NPC is currently interacting with, or None if the NPC is not interacting with any player.
    """
    def __init__(self, position=(-1, -1), health=100):
        super().__init__(position, health)
        self.inventory = []
        self.interacting_with = None
        

class PlayerManager(AgentManager):
    """Manages players in the game world.
    
    Attributes:
        players (list): A list of players in the game world.
    """
    def __init__(self):
        self.players = []
        
    def add_player(self, player):
        super().add_agent(player)
        
    def remove_player(self, player):
        super().remove_agent(player)
        
    def get_npcs_in_range(self, agent, range):
        npcs = super().get_agents_in_range(agent, range)
        return [npc for npc in npcs if isinstance(npc, NPC)]
        
    def print_player_info(self, player):
        print("Player position: ", player.position)
        print("Player health: ", player.health)
        print("Player inventory: ", player.inventory)
        print()
        

class NPCManager(AgentManager):
    """Manages NPCs in the game world.
    
    Attributes:
        npcs (list): A list of NPCs in the game world.
    """
    def __init__(self):
        self.npcs = []
        
    def add_npc(self, npc):
        super().add_agent(npc)
        
    def remove_npc(self, npc):
        super().remove_agent(npc)
        
    def print_npc_info(self, npc):
        print("NPC position: ", npc.position)
        print("NPC health: ", npc.health)
        print("NPC inventory: ", npc.inventory)


class Gameworld:
    """Represents the game world and manages the game map and agents.
    
    Attributes:
        map (list): A 2D list representing the game map, with 0 representing an empty cell and a player or NPC object representing a cell occupied by that player or NPC.
        player_manager (PlayerManager): A PlayerManager instance to manage players in the game world.
        npc_manager (NPCManager): An NPCManager instance to manage NPCs in the game world.
    """
    def __init__(self):
        self.map = [[0 for i in range(10)] for j in range(10)]
        self.player_manager = PlayerManager()
        self.npc_manager = NPCManager()
        
    def move_agent(self, agent, dx, dy):
        """Move an agent on the map by changing its position attribute and updating the map list.
        
        Args:
            agent (Agent): The agent to move.
            dx (int): The change in x position.
            dy (int): The change in y position.
            
        Returns:
            bool: True if the agent was successfully moved, False otherwise.
        """
        if 0 <= agent.x + dx < len(self.map) and 0 <= agent.y + dy < len(self.map[0]):
            if self.map[agent.x + dx][agent.y + dy] is not None:
                agent.position = (agent.x + dx, agent.y + dy)
                self.map[agent.x][agent.y] = agent
                self.map[agent.x - dx][agent.y - dy] = 0
                return True
        return False
    
    def print_map(self):
        """Print a representation of the game map.
        """
        for row in self.map:
            for cell in row:
                if cell == 0:
                    print(" ", end="")
                elif cell in self.player_manager.players:
                    print("P", end="")
                elif cell in self.npc_manager.npcs:
                    print("N", end="")
                    
            print()
            

def simulate_gameworld(steps, num_players, num_npcs):
    """Simulate a game world for a certain number of steps with a specified number of players and NPCs.
    
    Args:
        steps (int): The number of steps to simulate.
        num_players (int): The number of players to include in the game world.
        num_npcs (int): The number of NPCs to include in the game world.
    """
    gameworld = Gameworld()
    
    # Add players and NPCs to the game world
    for i in range(num_players):
        player = Player((random.randint(0, 9), random.randint(0, 9)))
        gameworld.player_manager.add_player(player)
        
    for i in range(num_npcs):
        npc = NPC((random.randint(0, 9), random.randint(0, 9)))
        gameworld.npc_manager.add_npc(npc)
        
    # Simulate the game world for the specified number of steps
    for i in range(steps):
        print(f"Step {i+1}:")
        
        # Move players and NPCs randomly
        for player in gameworld.player_manager.players:
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            gameworld.move_agent(player, dx, dy)
            
        for npc in gameworld.npc_manager.npcs:
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            gameworld.move_agent(npc, dx, dy)
            
        # Have players interact with NPCs if they are within range
        for player in gameworld.player_manager.players:
            player.interaction_with_npc(gameworld.player_manager.get_agents_in_range(player, 1))
            
        # Print information about the players and NPCs
        for player in gameworld.player_manager.players:
            gameworld.player_manager.print_player_info(player)
            
        for npc in gameworld.npc_manager.npcs:
            gameworld.npc_manager.print_npc_info(npc)
        
        print()


simulate_gameworld(steps=10, num_players=5, num_npcs=5)

"""
This revised code separates the game world and its agents into separate classes and introduces a player manager and NPC manager to handle the players and NPCs in the game world. The Gameworld class has methods to move agents, get agents within a certain range of an agent, and print a representation of the game map. The Player and NPC classes both inherit from the Agent class and have additional attributes and methods specific to their roles in the game. The PlayerManager and NPCManager classes have methods to add and remove players and NPCs from the game world, respectively.

You can use this code to create and manipulate a game world with players and NPCs. You can add or remove players and NPCs from the game world using the add_player and add_npc methods of the PlayerManager and NPCManager classes, respectively. You can move agents on the map using the move_agent method of the Gameworld class. You can also interact with NPCs and perform actions such as trading or fighting using the methods of the Player class. Finally, you can print a representation of the game map using the print_map method of the Gameworld class.
"""