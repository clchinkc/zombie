class Individual:
    def __init__(self, id):
        self.id = id
        self.connections = []
    
    def add_connection(self, other):
        self.connections.append(other)
        
    def get_info(self):
        return f'Individual {self.id} has {len(self.connections)} connections.'

class Population:
    def __init__(self, size):
        self.members = [Individual(id) for id in range(size)]
    
    def connect(self, individual1, individual2):
        individual1.add_connection(individual2)
        individual2.add_connection(individual1)
    
    def get_info(self):
        connection_info = []
        for individual in self.members:
            connection_info.append(f'Individual {individual.id} is connected to {[other.id for other in individual.connections]}')
        return f'Population of size {len(self.members)}' + '\n' + '\n'.join(connection_info)

pop = Population(1000)

# Connect a small number of individuals in the population
pop.connect(pop.members[0], pop.members[1])
pop.connect(pop.members[1], pop.members[2])
pop.connect(pop.members[0], pop.members[2])

# Print information about the population and the first individual
print(pop.get_info())
print(pop.members[0].get_info())
