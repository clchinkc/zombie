import math
import random
    
class Entity:
    """Represents a game world entity with a position and health.
    
    Attributes:
        position (tuple): The position of the entity on the map.
        health (int): The health of the entity.
    """
    def __init__(self, position, health=100):
        self.position = position
        self.health = health
        
    def distance_to_entity(self, other_entity):
        """Calculate the distance between this entity and another entity.
    
        Args:
            other_entity (Entity): The other entity to calculate the distance to.
    
        Returns:
            float: The distance between the two entities.
        """
        x1, y1 = self.position
        x2, y2 = other_entity.position
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

class EntityManager:
    """Manages entities in the game world.

    Attributes:
        entities (list): A list of entities in the game world.
    """
    def __init__(self):
        self.entities = []
    
    def add_entity(self, entity):
        """Add an entity to the game world.
    
        Args:
            entity (Entity): The entity to add.
        """
        self.entities.append(entity)
    
    def remove_entity(self, entity):
        """Remove an entity from the game world.
    
        Args:
            entity (Entity): The entity to remove.
        """
        self.entities.remove(entity)
    
    def get_entities_in_range(self, entity, range):
        """Get a list of entities within a certain range of a given entity.
    
        Args:
            entity (Entity): The entity to check the range from.
            range (float): The range to check for entities within.
        
        Returns:
            list: A list of entities within the specified range.
        """
        in_range = []
        for other_entity in self.entities:
            if entity.distance_to_entity(other_entity) <= range:
                in_range.append(other_entity)
        return in_range

class Player(Entity):
    """Represents a player in the game world. Inherits from Entity.
    
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


class NPC(Entity):
    """Represents an NPC in the game world. Inherits from Entity.
    
    Attributes:
        inventory (list): A list of items in the NPC's inventory.
        interacting_with (Player): The player that the NPC is currently interacting with, or None if the NPC is not interacting with any player.
    """
    def __init__(self, position=(-1, -1), health=100):
        super().__init__(position, health)
        self.inventory = []
        self.interacting_with = None
        

class PlayerManager(EntityManager):
    """Manages players in the game world.
    
    Attributes:
        players (list): A list of players in the game world.
    """
    def __init__(self):
        self.players = []
        
    def add_player(self, player):
        super().add_entity(player)
        
    def remove_player(self, player):
        super().remove_entity(player)
        
    def get_npcs_in_range(self, entity, range):
        npcs = super().get_entities_in_range(entity, range)
        return [npc for npc in npcs if isinstance(npc, NPC)]
        
    def print_player_info(self, player):
        print("Player position: ", player.position)
        print("Player health: ", player.health)
        print("Player inventory: ", player.inventory)
        print()
        

class NPCManager(EntityManager):
    """Manages NPCs in the game world.
    
    Attributes:
        npcs (list): A list of NPCs in the game world.
    """
    def __init__(self):
        self.npcs = []
        
    def add_npc(self, npc):
        super().add_entity(npc)
        
    def remove_npc(self, npc):
        super().remove_entity(npc)
        
    def print_npc_info(self, npc):
        print("NPC position: ", npc.position)
        print("NPC health: ", npc.health)
        print("NPC inventory: ", npc.inventory)


class Gameworld:
    """Represents the game world and manages the game map and entities.
    
    Attributes:
        map (list): A 2D list representing the game map, with 0 representing an empty cell and a player or NPC object representing a cell occupied by that player or NPC.
        player_manager (PlayerManager): A PlayerManager instance to manage players in the game world.
        npc_manager (NPCManager): An NPCManager instance to manage NPCs in the game world.
    """
    def __init__(self):
        self.map = [[0 for i in range(10)] for j in range(10)]
        self.player_manager = PlayerManager()
        self.npc_manager = NPCManager()
        
    def move_entity(self, entity, dx, dy):
        """Move an entity on the map by changing its position attribute and updating the map list.
        
        Args:
            entity (Entity): The entity to move.
            dx (int): The change in x position.
            dy (int): The change in y position.
            
        Returns:
            bool: True if the entity was successfully moved, False otherwise.
        """
        if 0 <= entity.x + dx < len(self.map) and 0 <= entity.y + dy < len(self.map[0]):
            if self.map[entity.x + dx][entity.y + dy] == 0:
                entity.position = (entity.x + dx, entity.y + dy)
                self.map[entity.x][entity.y] = entity
                self.map[entity.x - dx][entity.y - dy] = 0
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
            gameworld.move_entity(player, dx, dy)
            
        for npc in gameworld.npc_manager.npcs:
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            gameworld.move_entity(npc, dx, dy)
            
        # Have players interact with NPCs if they are within range
        for player in gameworld.player_manager.players:
            player.interaction_with_npc(gameworld.player_manager.get_entities_in_range(player, 1))
            
        # Print information about the players and NPCs
        for player in gameworld.player_manager.players:
            gameworld.player_manager.print_player_info(player)
            
        for npc in gameworld.npc_manager.npcs:
            gameworld.npc_manager.print_npc_info(npc)
        
        print()


simulate_gameworld(steps=10, num_players=5, num_npcs=5)

"""
This revised code separates the game world and its entities into separate classes and introduces a player manager and NPC manager to handle the players and NPCs in the game world. The Gameworld class has methods to move entities, get entities within a certain range of an entity, and print a representation of the game map. The Player and NPC classes both inherit from the Entity class and have additional attributes and methods specific to their roles in the game. The PlayerManager and NPCManager classes have methods to add and remove players and NPCs from the game world, respectively.

You can use this code to create and manipulate a game world with players and NPCs. You can add or remove players and NPCs from the game world using the add_player and add_npc methods of the PlayerManager and NPCManager classes, respectively. You can move entities on the map using the move_entity method of the Gameworld class. You can also interact with NPCs and perform actions such as trading or fighting using the methods of the Player class. Finally, you can print a representation of the game map using the print_map method of the Gameworld class.
"""