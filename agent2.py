# model/simulate a human in a zombie apocalypse
# focus on movement and health

# import the necessary packages
import numpy as np
import random
import matplotlib.pyplot as plt

# define the class for the human
class Human:
    def __init__(self, x, y, health, strength, speed, status):
        # initialize the human

    # update the human
    def update(self, humans, zombies):
        # attack the zombie
        # check if the human is infected
        # check if the human is dead
        # move the human
                
    def attack(self, target):
        # if the target is a zombie, fight
            # if the human is faster than the zombie, they hit first
                # the zombie's health is reduced by the human's strength
                # if the zombie is still alive, attack back
                    # the human's health is reduced by the zombie's strength
                    # if the human is dead, turn to zombie
                # if the zombie is dead, die
            # if the zombie is faster than the human, they hit first
                # the human's health is reduced by the zombie's strength
                # if the human is still alive, attack back
                    # the zombie's health is reduced by the human's strength
                    # if the zombie is dead, die
                # if the human is dead, turn to zombie

    # move the human
    def move(self, humans, zombies):
        # get the possible moves
        # check if the human can move
        # check if the human is next to a zombie
            # move away from the zombie
        # check if the human is next to another human
            # move towards the other human
        # move the human by dx and dy

            
    def fleet(self, humans, zombies):
        # check if the human is next to a zombie
            # move away from the zombie
        # check if the human is next to another human
        
    def __str__(self):
        return str(self.status) + 'human with ' + str(self.health) + ' health, ' + str(self.strength) + ' strength, and ' + str(self.speed) + ' speed'

        
# define the class for the zombie
class Zombie:
    def __init__(self, x, y, health, strength, speed):
        # initialize the zombie

    # update the zombie
    def update(self, humans, zombies):
        # update the zombie's health
        # check if the zombie is dead
        # move the zombie
    
    # attack the human        
    def attack(self, target):
        # if the target is a human, fight
            # if the zombie is faster than the human, they hit first
                # the human's health is reduced by the zombie's strength
                # if the human is still alive, attack back
                    # the zombie's health is reduced by the human's strength
                    # if the zombie is dead, delete the zombie
                # if the human is dead, turn to zombie
            # if the human is faster than the zombie, they hit first
                # the zombie's health is reduced by the human's strength
                # if the zombie is still alive, attack back
                    # the human's health is reduced by the zombie's strength
                    # if the human is dead, turn to zombie
                # if the zombie is dead, delete the zombie

    # move the zombie
    def move(self, humans, zombies):
        # get the possible moves
        # check if the zombie can move
        # check if the zombie is next to a zombie
        # check if the zombie is next to a human
        # move the zombie by dx and dy
        
    def __str__(self):
        return 'zombie with ' + str(self.health) + ' health, ' + str(self.strength) + ' strength, and ' + str(self.speed) + ' speed'

    
# determine the behaviour and interaction of the human and zombie classes  
# define the class for the apocalypse
class Apocalypse:
    def __init__(self, humans, zombies):
        # initialize the apocalypse

    # run the apocalypse
    def run(self, iterations):
        # initialize the lists
        # loop over the number of iterations
            # update the apocalypse
            # count the number of zombies and humans
            # if there are no humans left, the zombies win
            # if there are no zombies left, the humans win
            # if there are still humans and zombies, the apocalypse continues
        # plot the results
        
    # initialize the simulation
    def intialize(self):
        # create a list of humans
        # create a list of zombies
        # create the humans in a random for loop
        # create the zombies in a random for loop
        # return the humans and zombies lists
        
    # update the apocalypse
    def update(self):
        # update the humans in a for loop
        # update the zombies
        # return the humans and zombies

    def count(self):
        # count the number of zombies and humans
        # return the human and zombie counts
    
    # plot the results
    def plot(self):
        # print the human and zombie counts
        # plot the results of human and zombie count to iterations

# create a simulation that run the model
def simulation():
    # initialize the humans list
    # initialize the zombies list
    # initialize the apocalypse
    # run the apocalypse
    # plot the results
    
# run the simulation

"""
human may attack human who is infected
human may pay some health to escape zombie within a range
"""