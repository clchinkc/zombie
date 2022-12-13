import numpy as np

# Define the size of the grid (i.e. the school)
ROWS = 10
COLS = 10

# Define the possible states of a cell in the grid
STUDENT = 0
TEACHER = 1
ZOMBIE = 2
EMPTY = 3

# Define the rules for transitioning between states
def transition(state, neighbors):
  # If the cell is a student, it becomes a zombie if it is surrounded by three or more zombie cells
  if state == STUDENT and np.sum(neighbors == ZOMBIE) >= 3:
    return ZOMBIE
  
  # If the cell is a teacher, it can move to an adjacent empty cell if it is surrounded by two or more student cells
  elif state == TEACHER and np.sum(neighbors == STUDENT) >= 2:
    return EMPTY
  
  # If the cell is a zombie, it is killed if it is surrounded by two or more student cells
  elif state == ZOMBIE and np.sum(neighbors == STUDENT) >= 2:
    return EMPTY
  
  # Otherwise, the cell remains in its current state
  else:
    return state

# Initialize the grid with random cell states
grid = np.random.randint(0, 4, size=(ROWS, COLS))

# Run the CA for 100 steps
for i in range(10):
  # Create a new empty grid to store the updated states
  new_grid = np.zeros((ROWS, COLS), dtype=int)
  
  # Iterate over the cells in the grid
  for row in range(ROWS):
    for col in range(COLS):
      # Get the state of the current cell and its neighbors
      state = grid[row, col]
      neighbors = grid[max(row - 1, 0):min(row + 2, ROWS), max(col - 1, 0):min(col + 2, COLS)]
      
      # Update the state of the cell based on its neighbors and the rules
      new_state = transition(state, neighbors)
      new_grid[row, col] = new_state
  
  # Update the grid with the new cell states
  grid = new_grid
  
  # Print the current state of the grid
  print(grid)

"""
One potential improvement to the CA model could be to add more complex rules for transitioning between states. 
For example, the model could incorporate additional factors 
such as the age or health of the students, teachers, and zombies, 
as well as the availability of weapons or other resources. 
This could allow for more realistic and nuanced simulations of the zombie apocalypse at school.

Another potential improvement could be to add more advanced visualization techniques, 
such as color-coding the cells to represent different states or using animation to show the progression of the simulation over time. 
This could make it easier for users to understand and interpret the results of the simulation.

Additionally, the model could be expanded to include more detailed information about the layout of the school, 
such as the locations of classrooms, doors, and other features. 
This could allow for more accurate simulations of the movement 
and interactions of students, teachers, and zombies within the school environment.
"""