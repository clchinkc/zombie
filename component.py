
from abc import ABC, abstractmethod


class InputComponent(ABC):
    @abstractmethod
    def update(self, obj):
        pass

class PlayerInputComponent(InputComponent):

    def __init__(self) -> None:
        self.WALK_ACCELERATION = 1

    def update(self, obj):
        if Controller.getJoystickDirection() == DIR_LEFT:
            obj.velocity -= self.WALK_ACCELERATION
        elif Controller.getJoystickDirection() == DIR_RIGHT:
            obj.velocity += self.WALK_ACCELERATION


class DemoInputComponent(InputComponent):
    # AI to automatically control Bjorn
    def update(self, obj):
        pass

class PhysicsComponent(ABC):
    @abstractmethod
    def update(self, obj, world):
        pass

class BjornPhysicsComponent(PhysicsComponent):
    
    def __init__(self, volume) -> None:
        self.volume = volume
    
    def update(self, obj, world):
        obj.x += obj.velocity
        world.resolveCollision(self.volume, obj.x, obj.y, obj.velocity)

class GraphicsComponent:
    def update(self, obj, graphics):
        pass

class BjornGraphicsComponent(GraphicsComponent):
    
    def __init__(self, spriteStand, spriteWalkLeft, spriteWalkRight) -> None:
        self.spriteStand = spriteStand
        self.spriteWalkLeft = spriteWalkLeft
        self.spriteWalkRight = spriteWalkRight
    
    def update(self, obj, graphics):
        if obj.velocity < 0:
            sprite = self.spriteWalkLeft
        elif obj.velocity > 0:
            sprite = self.spriteWalkRight
        else:
            sprite = self.spriteStand
        graphics.draw(sprite, obj.x, obj.y)

class GameObject:
    def __init__(self, input, physics, graphics):
        self.velocity = 0
        self.x = 0
        self.y = 0
        self.input = input
        self.physics = physics
        self.graphics = graphics

    def update(self, world, graphics):
        self.input.update(self)
        self.physics.update(self, world)
        self.graphics.update(self, graphics)

def createBjorn():
    return GameObject(PlayerInputComponent(), 
                        BjornPhysicsComponent(), 
                        BjornGraphicsComponent())

"""
User inputcomponent that has method using Character as input to update the Character
Update the inputcomponent inside the Character update method
The Character own the inputcomponent that control the Character velocity by a specific acceleration stored inside inputcomponent
The character also own a physicscomponent that change the Character position according to its velocity and world.resolve collision function with the character and world as input
The character also own a graphicscomponent that points to left if velocity lower than 0 else right and graphic.draw function with character and graphics as input
So the character has position, velocity, inputcomponent, physicscomponent, graphicscomponent and update function that call input.update, physics.update, graphics.update
It holds components and state shared among components
"""

"""
How do components communicate with each other?

By modifying the container object's state: shared state in parent object

By referring directly to each other: but coupled

By sending messages: fire-and-forget messaging component in each component
"""
"""
Unsurprisingly, there's no one best answer here. What you'll likely end up doing is using a bit of all of them. Shared state is useful for the really basic stuff that you can take for granted that every object has — things like position and size.

Some domains are distinct but still closely related. Think animation and rendering, user input and AI, or physics and collision. If you have separate components for each half of those pairs, you may find it easiest to just let them know directly about their other half.

Messaging is useful for “less important” communication. Its fire-and-forget nature is a good fit for things like having an audio component play a sound when a physics component sends a message that the object has collided with something.

As always, I recommend you start simple and then add in additional communication paths if you need them.
"""
"""
https://gameprogrammingpatterns.com/component.html
# https://docs.gamecreator.io/gamecreator/characters/component/
"""
