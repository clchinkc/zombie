
import sys
import time

import numpy as np
import pygame
from pygame.locals import *

# window size
WIDTH = 80
HEIGHT = 40


# global variables to record mouse button status
button_down = False
button_type = None

# game world matrix
world = np.zeros((HEIGHT, WIDTH))

# Create Cell class to represent each cell in the game of life
class Cell(pygame.sprite.Sprite):

    size = 10

    def __init__(self, position):

        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface([self.size, self.size])

        # set the color of the cell to white
        self.image.fill((255, 255, 255))

        # create a rectangle with the left top corner as the anchor point
        self.rect = self.image.get_rect()
        self.rect.topleft = position

    def draw(self, screen):
        screen.blit(self.image, self.rect)


# draw function, note that we reset the screen and then traverse the whole world map
# create a list of cell objects
cell_objs = [[Cell((col * Cell.size, row * Cell.size)) for col in range(WIDTH)] for row in range(HEIGHT)]

def draw(screen):
    screen.fill((0, 0, 0))
    for sp_col in range(world.shape[1]):
        for sp_row in range(world.shape[0]):
            if world[sp_row][sp_col]:
                cell_objs[sp_row][sp_col].draw(screen)



# update the game world according to the rules of the game of life
def next_generation():
    global world
    nbrs_count = sum(
        np.roll(np.roll(world, i, 0), j, 1)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        if (i != 0 or j != 0)
    )

    world = (nbrs_count == 3) | ((world == 1) & (nbrs_count == 2)).astype("int")


# initialize the game
def init():
    global world
    world.fill(0)
    draw(screen)
    return "Stop"


# stop function, we can use the mouse to draw the initial state of the game world
def stop():
    global button_down, button_type
    for event in pygame.event.get():
        if event.type == QUIT: # press the close button to exit
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN and event.key == K_RETURN: # press enter to start
            return "Move"

        if event.type == KEYDOWN and event.key == K_r: # press r to reset
            return "Reset"

        if event.type == MOUSEBUTTONDOWN: # press mouse button to draw
            button_down = True
            button_type = event.button

        if event.type == MOUSEBUTTONUP: # release mouse button
            button_down = False

        if button_down: # draw
            mouse_x, mouse_y = pygame.mouse.get_pos()

            sp_col = mouse_x // Cell.size
            sp_row = mouse_y // Cell.size

            if button_type == 1:  # left mouse button
                world[sp_row][sp_col] = 1
            elif button_type == 3:  # right mouse button
                world[sp_row][sp_col] = 0
            draw(screen)

    return "Stop"

# timer, control frame rate
clock_start = 0

# move function, we can use the space key to pause and the r key to reset
def move():
    global button_down, button_type, clock_start
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN and event.key == K_SPACE:
            return "Stop"
        if event.type == KEYDOWN and event.key == K_r:
            return "Reset"
        if event.type == MOUSEBUTTONDOWN:
            button_type = event.button
            button_down = True
        if event.type == MOUSEBUTTONUP:
            button_down = False

        if button_down:
            mouse_x, mouse_y = pygame.mouse.get_pos()

            sp_col = mouse_x // Cell.size
            # use integer division
            sp_row = mouse_y // Cell.size

            if button_type == 1:
                world[sp_row][sp_col] = 1
            elif button_type == 3:
                world[sp_row][sp_col] = 0
            draw(screen)
        
    if time.time() - clock_start > 0.1: # control frame rate
        clock_start = time.time()
        next_generation()
        draw(screen)

    return "Move"


if __name__ == "__main__":

    # state machine corresponds to three states, initialization, stop, and proceed
    state_actions = {"Reset": init, "Stop": stop, "Move": move}
    state = "Reset"

    pygame.init()
    pygame.display.set_caption("Conway's Game of Life")

    screen = pygame.display.set_mode((WIDTH * Cell.size, HEIGHT * Cell.size))
    clock = pygame.time.Clock()
    fps = 30  # set desired frame rate

    while True:  # main loop

        state = state_actions[state]()
        pygame.display.update()
        clock.tick(fps)  # control frame rate

"""
# https://bbs.huaweicloud.com/blogs/138603
# https://cloud.tencent.com/developer/article/1552874
# https://www.lanqiao.cn/courses/769
# https://fireholder.github.io/posts/lifeGame
# https://www.codercto.com/a/57167.html
# https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
"""