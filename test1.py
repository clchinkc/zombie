


class Myclass:
    def __init__(self):
        self.x = 1
        self.y = 2
        self.z = 3

    def print(self) -> str:
        info = f"{self.__class__.__name__}\n"
        for var in vars(self):
            info += f"{var}: {getattr(self, var)}\n"
        return info
    
    def __repr__(self):
        return "{classname}({variables})".format(classname=self.__class__.__name__, variables=", ".join([str(getattr(self, var)) for var in vars(self)]))

    def __str__(self):
        return f"Myclass {self.x} {self.y} {self.z}"
    

newclass = Myclass()
print(newclass.print())
print(newclass.__repr__())
print(newclass.__str__())
