import pickle

class PickleStorage(object):
    def __init__(self, filename):
        self.filename = filename

    def save(self, data):
        with open(self.filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        with open(self.filename, 'rb') as f:
            return pickle.load(f)
        
        
# example code to manupilate a data structure

data = PickleStorage('data.pickle').load()
# do something with data
PickleStorage('data.pickle').save(data)




