
import random
import sys

import pygame

# Initialize Pygame
pygame.init()

# Pygame variables
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 50
ENEMY_WIDTH = 50
ENEMY_HEIGHT = 50
PLATFORM_WIDTH = 100
PLATFORM_HEIGHT = 20
COIN_RADIUS = 30


# Game variables
PLAYER_SPEED = 5
PLAYER_JUMP_FORCE = 15
ENEMY_SPEED = 3
PLATFORM_POSITIONS = [(0, WINDOW_HEIGHT - 20), (100, WINDOW_HEIGHT - 20),
                      (200, WINDOW_HEIGHT - 20), (300, WINDOW_HEIGHT - 20),
                      (400, WINDOW_HEIGHT - 20), (500, WINDOW_HEIGHT - 20),
                      (600, WINDOW_HEIGHT - 20), (700, WINDOW_HEIGHT - 20),
                      (300, 450), (600, 300)]


# Initialize the game window with background image
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
background_image = pygame.image.load("./platformer/background.png").convert_alpha()
background_image = pygame.transform.scale(background_image, (WINDOW_WIDTH, WINDOW_HEIGHT))


# Load images
player_image = pygame.image.load("./platformer/player.png").convert_alpha()
enemy_image = pygame.image.load("./platformer/enemy.png").convert_alpha()
coin_image = pygame.image.load("./platformer/coin.png").convert_alpha()
platform_image = pygame.image.load("./platformer/platform.png").convert_alpha()

# Scale images
scaled_player_image = pygame.transform.scale(player_image, (PLAYER_WIDTH, PLAYER_HEIGHT))
scaled_enemy_image = pygame.transform.scale(enemy_image, (ENEMY_WIDTH, ENEMY_HEIGHT))
scaled_coin_image = pygame.transform.scale(coin_image, (COIN_RADIUS * 2, COIN_RADIUS * 2))
scaled_platform_image = pygame.transform.scale(platform_image, (PLATFORM_WIDTH, PLATFORM_HEIGHT))


# Helper functions
def draw_text(text, font, color, x, y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = (x, y)
    window.blit(text_surface, text_rect)

def check_collision(rect1, rect2):
    return rect1.colliderect(rect2)

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = scaled_player_image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.vel_y = 0
        self.vel_x = 0
        self.on_ground = False
        self.lives = 3
        self.invulnerable = False
        self.invulnerability_timer = 0

    def update(self):
        self.vel_y += 0.5  # Always apply gravity, no matter what
        self.on_ground = False  # Assume we are in the air unless we hit something

        # Check for horizontal collisions first
        future_rect = self.rect.copy()
        future_rect.x += self.vel_x
        self.check_platform_collision(future_rect)

        # Apply horizontal movement
        self.rect.x += self.vel_x

        # Check for vertical collisions
        future_rect = self.rect.copy()
        future_rect.y += self.vel_y
        self.check_platform_collision(future_rect)

        # Apply vertical movement
        self.rect.y += self.vel_y

        self.check_coin_collision()
        self.check_enemy_collision()
        self.handle_invulnerability()

    def check_platform_collision(self, future_rect):
        # Check for collisions with platforms
        for platform in platforms:
            if future_rect.colliderect(platform.rect):
                # Check if the player is above the platform
                if self.rect.bottom <= platform.rect.top:
                    self.rect.bottom = platform.rect.top
                    self.vel_y = 0
                    self.on_ground = True  # Set on_ground to True only if there is a collision
                # Check if the player is below the platform
                elif self.rect.top >= platform.rect.bottom:
                    self.rect.top = platform.rect.bottom
                    self.vel_y = 0
                # Check if the player is on the left side of the platform
                elif self.rect.right <= platform.rect.left:
                    self.rect.right = platform.rect.left
                    self.vel_x = 0
                # Check if the player is on the right side of the platform
                elif self.rect.left >= platform.rect.right:
                    self.rect.left = platform.rect.right
                    self.vel_x = 0

    def gravity(self):
        if not self.on_ground:
            self.vel_y += 0.5
        if self.rect.y >= WINDOW_HEIGHT - self.rect.height and self.vel_y >= 0:
            self.rect.y = WINDOW_HEIGHT - self.rect.height
            self.vel_y = 0
            self.on_ground = True
        self.rect.y += self.vel_y

    def jump(self):
        if self.on_ground:
            self.vel_y -= PLAYER_JUMP_FORCE

    def check_coin_collision(self):
        # Check for collisions with coins
        coins_collected = pygame.sprite.spritecollide(self, coins, True)
        for coin in coins_collected:
            # Increase score
            score.increment()
        if len(coins) == 0:
            end_game("YOU WIN!")

    def check_enemy_collision(self):
        # Check for collisions with enemies
        if pygame.sprite.spritecollideany(self, enemies) and not self.invulnerable:
            self.lose_life()

    def lose_life(self):
        if self.lives > 0:
            self.lives -= 1
            lives.decrement()
            self.invulnerable = True
            self.invulnerability_timer = pygame.time.get_ticks()
            if self.lives == 0:
                end_game("GAME OVER")

    def handle_invulnerability(self):
        if self.invulnerable and pygame.time.get_ticks() - self.invulnerability_timer > 1000:
            self.invulnerable = False

class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = scaled_enemy_image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.vel_x = ENEMY_SPEED

    def update(self):
        self.rect.x += self.vel_x
        if self.rect.left < 0 or self.rect.right > WINDOW_WIDTH:
            self.vel_x *= -1

class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = scaled_platform_image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Coin(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = scaled_coin_image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Score:
    def __init__(self):
        self.value = 0
        self.font = pygame.font.Font(None, 36)

    def increment(self):
        self.value += 1

    def draw(self):
        draw_text("Score: " + str(self.value), self.font, (255, 255, 255), 70, 50)

class Lives:
    def __init__(self):
        self.value = 3
        self.font = pygame.font.Font(None, 36)

    def decrement(self):
        self.value -= 1

    def draw(self):
        draw_text("Lives: " + str(self.value), self.font, (255, 255, 255), WINDOW_WIDTH - 70, 50)


def create_game_objects():
    player = Player(100, 100)
    platforms = create_platforms()
    coins = create_coins()
    enemies = create_enemies()
    score = Score()
    lives = Lives()
    
    return player, platforms, coins, enemies, score, lives

def create_platforms():
    platforms = pygame.sprite.Group()
    for pos in PLATFORM_POSITIONS:
        platforms.add(Platform(*pos))
    return platforms

def create_coins():
    coins = pygame.sprite.Group()
    for _ in range(10):
        x = random.randint(100, 700)
        y = random.randint(100, 400)
        coins.add(Coin(x, y))
    return coins

def create_enemies():
    enemies = pygame.sprite.Group()
    enemies.add(Enemy(400, 500))
    enemies.add(Enemy(700, 250))
    return enemies


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                player.jump()
            if event.key == pygame.K_LEFT:
                player.vel_x = -PLAYER_SPEED
            if event.key == pygame.K_RIGHT:
                player.vel_x = PLAYER_SPEED
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT and player.vel_x < 0:
                player.vel_x = 0
            if event.key == pygame.K_RIGHT and player.vel_x > 0:
                player.vel_x = 0
    return True


def end_game(text):
    draw_text(text, pygame.font.Font(None, 48), (255, 255, 255), WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    pygame.display.update()
    pygame.time.delay(3000)
    pygame.quit()


clock = pygame.time.Clock()
player, platforms, coins, enemies, score, lives = create_game_objects()

running = True
while running:
    running = handle_events()

    window.fill((0, 0, 0))
    window.blit(background_image, (0, 0))

    player.update()
    enemies.update()

    window.blit(player.image, player.rect)
    enemies.draw(window)
    platforms.draw(window)
    coins.draw(window)
    score.draw()
    lives.draw()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()


# The character can climb ladders to go to the platform too.
# Add a sound effect when the character collects a coin or loses a life or jumps.
# divide the code into different modules and import them into the main file, main.py. player.py, enemy.py, platform.py, physics.py, input.py, sound.py
# Add a health bar for the player character to replace the lives system.

"""
1. Conceptualize the game's overall design, mechanics, theme, and art style.
2. Design simple and stylized 2D graphics for all the sprites.
3. Design and create sprites for the player characters jumping animation.
4. Develop the player characters jumping mechanics and animations.
5. Implement gravity and falling mechanics for the player character.
6. Create level designs with increasing difficulty, unique obstacles, and challenges for the player to overcome.
7. Implement a camera system to follow the player character.
8. Create enemy character sprites and implement their movement and behavior.
9. Implement enemy AI and behavior to make them more challenging to defeat.
10. Add collectible items such as coins and power-ups to the game world.
11. Add powerups and special abilities such as invincibility or increased speed for the player character.
12. Create a health system for the player character and enemies.
13. Create boss battles for each level that require unique strategies to defeat.
14. Implement scoring and game over conditions.
15. Implement a scoring system that tracks the players progress, rewards the player for collecting coins, defeating enemies, and achievements.
16. Create a tutorial or instructions for new players to learn the game mechanics.
17. Add sound effects and background music to enhance the player's experience.
18. Add additional sound effects and music tracks to enhance the player's experience.
19. Implement a save and load system for player progress.
20. Add multiplayer functionality to allow players to compete or cooperate with each other.
21. Create a level editor to design and save levels.
22. Implement online features and networking for multiplayer and leaderboard functionality.
23. Create a leaderboard to track high scores and achievements.
24. Test and debug the game to ensure it runs smoothly on different platforms and devices.
25. Optimize game performance for smoother gameplay on lower-end devices.
26. Create a user-friendly interface, including menus and settings.
27. Handle user inputs for various devices (touchscreen, keyboard/mouse, gamepad).
28. Develop in-game purchase systems if necessary.
29. Implement analytics to track player behavior for future improvements.
30. Set up an effective feedback system to allow players to report bugs and provide suggestions.
31. Carry out extensive beta testing to gather feedback and identify areas for improvement.
32. Publish and distribute the game to the desired platform such as a website or app store.
33. Set up post-launch support for updates, bug fixes, and added content.
34. Market the game through various channels to increase visibility and attract players.
"""
