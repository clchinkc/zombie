import math

class Human:
    def __init__(self, name):
        self.name = name
        self.position = (0, 0)
        self.health = 100
        
    def move(self, x, y):
        self.position = (self.position[0] + x, self.position[1] + y)
        return self.position
    
    def euclidean_distance(self, other):
        x1, y1 = self.position
        x2, y2 = other.position
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def attack(self, other):
        if self.euclidean_distance(other) <= 3:
            other.health -= 10
            return f"{self.name}:{self.position} attacked {other.name}:{other.position}. \
                    {other.name}'s health is now {other.health}"

class SurvivalGame:
    def __init__(self, map_size, resources, dangers):
        self.map_size = map_size
        self.resources = resources
        self.dangers = dangers
        self.players = []
        
    def add_player(self, player):
        self.players.append(player)
        
    def get_resource(self, player):
        if player.position in self.resources:
            self.resources.remove(player.position)
            return f"{player.name}:{player.position} found some resources."
        return "No resources found."
    
    def get_hurt(self, player):
        if player.position in self.dangers:
            player.health -=10
            return f"{player.name}:{player.position} is in dangers. \
                    {player.name}'s health is now {player.health}."
        return f"{player.name}:{player.position} is safe now."
    
    def get_state(self):
        return "resources: " + str(self.resources) + "\n" + "dangers: " + str(self.dangers) + "\n"

"""
game = SurvivalGame(10, [(1, 2), (3, 4), (5, 6)], [(7, 8), (9, 0)])
game.move(1, 0)
print (game.human_position)
game.move(0, 2)
print (game.human_position)
print(game.get_hurt())
print(game.get_resource())
print(game.get_state())
"""
