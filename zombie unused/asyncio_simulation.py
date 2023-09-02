
import asyncio
import random


class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [[None for _ in range(width)] for _ in range(height)]
        self.locks = [[asyncio.Lock() for _ in range(width)] for _ in range(height)]

class Entity:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
    
    async def move_to(self, new_x, new_y, grid, max_retries=3):
        # Check if the move is only one step in one of the specified directions
        if (new_x, new_y) not in [(self.x, self.y + 1), (self.x + 1, self.y), (self.x, self.y - 1), (self.x - 1, self.y)]:
            print(f"{self.name} tried to move in an invalid direction!")
            return
        
        for _ in range(max_retries):
            current_lock = grid.locks[self.y][self.x]
            target_lock = grid.locks[new_y][new_x]
            
            try:
                acquired = await asyncio.wait_for(target_lock.acquire(), timeout=0.2)
                if acquired:
                    if grid.cells[new_y][new_x] is None:
                        grid.cells[self.y][self.x] = None
                        await asyncio.sleep(0.1)
                        self.x, self.y = new_x, new_y
                        grid.cells[new_y][new_x] = self
                        print(f"{self.name} moved to ({new_x}, {new_y})")
                        target_lock.release()
                        return
                    target_lock.release()
            except asyncio.TimeoutError:
                pass
                
            await asyncio.sleep(0.1)  # Wait before retrying

        # If movement fails, move to a nearby location
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                await self.move_to(nx, ny, grid, max_retries=1)
                return

        print(f"{self.name} couldn't find a place to move!")

async def main():
    grid = Grid(5, 5)
    
    entity1 = Entity("Entity1", 1, 2)
    entity2 = Entity("Entity2", 3, 2)
    
    grid.cells[1][2] = entity1
    grid.cells[3][2] = entity2

    # Both entities try to move to (2, 2).
    await asyncio.gather(entity1.move_to(2, 2, grid), entity2.move_to(2, 2, grid))

asyncio.run(main())


"""
Additional Considerations:
You might want to introduce randomness in movement decisions to make the simulation more dynamic.
If there are many entities and frequent movements, you might encounter a situation known as "deadlock". It's a state where entities are waiting for each other to release locks indefinitely. Design your logic carefully to avoid such situations or consider adding timeouts to the locks.
Other synchronization primitives like asyncio.Semaphore or asyncio.Condition can be used depending on your specific requirements.
"""