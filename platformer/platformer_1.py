# Import the pygame library
import pygame

# Initialize pygame
pygame.init()

# Set the screen size
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Set the game title
pygame.display.set_caption("My Platformer Game")

# Set the background color
background_color = (255, 255, 255)

# Set the player properties
player_width = 50
player_height = 50
player_color = (0, 0, 255)
player_x = 50
player_y = 50
player_speed = 5

# Set the game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(background_color)

    # Draw the player
    pygame.draw.rect(screen, player_color, (player_x, player_y, player_width, player_height))

    # Update the screen
    pygame.display.update()

# Quit pygame
pygame.quit()


"""
The most common features of platformer games are:
- Jumping challenges
- Completing levels
- Collecting items
- Avoiding obstacles
- Boss battles
- Power-ups
- Hidden areas
- Multiple paths
- Platforming mechanics (e.g. wall jumping, double jumping)
- Storyline
- Character customization
- Multiplayer modes
"""

"""
The platformer game will be designed using the following features:
- Use of the Sprite class and Group classes to organize and efficiently manage game objects
- Collision functions to detect intersecting bounding rectangles between sprites in different groups
- Maintaining the order of sprites for rendering
- Handling layers and draws
- Adding and removing sprites from a group
- Returning an ordered list of sprites
- Drawing all sprites in the right order
- Returning a list with all sprites at a position
- Returning the sprite at a specific index
- Removing all sprites from a layer and returning them as a list
- Returning a list of layers defined
- Changing the layer of a sprite
- Returning the layer that a sprite is currently in
- Returning the top and bottom layers
- Bringing a sprite to the front layer
- Moving a sprite to the bottom layer
- Returning the topmost sprite
- Returning all sprites from a layer ordered by how they were added
- Switching sprites from one layer to another
- Collision detection between sprites using rects or circles, with options for scaling the rects and using masks
"""

"""
Feature,Description
Endless runners,Games that take movement out of the player's hands so you can just focus on jumping.
Challenging platforming,Games that have challenging platforming that requires precision and timing.
Unique experiences,Games that offer unique experiences that are not found in other platformer games.
Creative settings,Games that have creative settings that allow for unique gameplay mechanics.
"""