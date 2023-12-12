import asyncio
import concurrent.futures
import random
import time


class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [[None for _ in range(width)] for _ in range(height)]
        self.locks = [[asyncio.Lock() for _ in range(width)] for _ in range(height)]

    def is_within_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

class Entity:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.override_decision = None  # Added for testing specific movements

    def is_valid_move(self, new_x, new_y):
        return (new_x, new_y) in [(self.x, self.y), (self.x, self.y + 1), (self.x + 1, self.y), (self.x, self.y - 1), (self.x - 1, self.y)]

    async def move_to(self, new_x, new_y, grid, max_retries=3):
        if not self.is_valid_move(new_x, new_y):
            print(f"{self.name} tried to move in an invalid direction!")
            return

        for _ in range(max_retries):
            if not grid.is_within_bounds(new_x, new_y):
                print(f"{self.name} tried to move out of bounds!")
                return

            async with grid.locks[self.x][self.y]:
                if grid.cells[new_x][new_y] is not None:
                    print(f"{self.name} found {grid.cells[new_x][new_y].name} at ({new_x}, {new_y})")
                    await asyncio.sleep(0.1)
                    continue

                if grid.locks[new_x][new_y].locked():
                    print(f"{self.name} found the lock at ({new_x}, {new_y}) is already acquired")
                    await asyncio.sleep(0.1)
                    continue

                async with grid.locks[new_x][new_y]:
                    grid.cells[self.x][self.y] = None
                    grid.cells[new_x][new_y] = self
                    self.x, self.y = new_x, new_y
                    print(f"{self.name} moved to ({new_x}, {new_y})")
                    return

        print(f"{self.name} couldn't find a place to move!")

    async def make_decision(self):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            decision = await loop.run_in_executor(pool, self.complex_decision_logic, self.x, self.y)
            return decision

    def complex_decision_logic(self, x, y):
        time.sleep(random.uniform(0.3, 0.5))  # Simulate delay
        if self.override_decision is not None:
            # Use the overridden decision for testing specific movements
            return self.override_decision
        while True:
            move_x = random.choice([True, False])
            step = random.choice([-1, 1])
            new_x = x + step if move_x else x
            new_y = y + step if not move_x else y
            # check if the new position is valid
            if self.is_valid_move(new_x, new_y):
                return new_x, new_y

    async def act(self, grid):
        new_x, new_y = await self.make_decision()
        await self.move_to(new_x, new_y, grid)

async def main():
    grid = Grid(5, 5)
    entity1 = Entity("Entity1", 1, 2)
    entity2 = Entity("Entity2", 2, 3)

    # Place entities on the grid
    grid.cells[1][2] = entity1
    grid.cells[2][3] = entity2

    # Both entities try to move to (2, 2), but only entity1 succeeds.
    entity1.override_decision = (2, 2)
    entity2.override_decision = (2, 2)
    # Move both entities to (2, 2) in sequence to ensure entity1 moves to (2, 2) first
    await entity1.act(grid)
    await entity2.act(grid)

    # After entity1 moves to (2, 3), entity2 can move to (2, 2).
    entity1.override_decision = None
    entity2.override_decision = (2, 2)
    # Move both entities in parallel to test whether entity2 can move to the original position of entity1
    await asyncio.gather(entity1.act(grid), entity2.act(grid))

asyncio.run(main())
