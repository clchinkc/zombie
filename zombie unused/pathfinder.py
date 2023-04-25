





"""
class maze:
	def __init__(self,layout,exits=None,name='Maze',time=0,random=None,end=None):
		self.layout=layout # a list of lists of numbers
		self.exits=exits # a list of the exits in the order of north, east, south, west
		self.name=name.capitalize() # the name of the maze
		self.time=time*60
		self.random=random # the size of the random maze
		self.end=end # the end message
	def get_maze(self):
		if self.random:
			return generate(self.random*4-1,self.up)
		else:
			return self.layout
	def get_exits(self):
		return self.exits

# master_maze a list of all the mazes in the game and some places storing special items
"""



# https://realpython.com/python-maze-solver/