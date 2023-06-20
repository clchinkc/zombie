
def zip_longest(iter1, iter2, fillvalue=None):
    
    for i in range(max(len(iter1), len(iter2))):
        if i >= len(iter1):
            yield (fillvalue, iter2[i])
        elif i >= len(iter2):
            yield (iter1[i], fillvalue)
        else:
            yield (iter1[i], iter2[i])
        i += 1

class Polynomial:

    def __init__(self, *coefficients):
        self.coefficients = list(coefficients)

    def __repr__(self):
        return "Polynomial" + str(tuple(self.coefficients))

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return self.coefficients == other.coefficients
        return NotImplemented

    def __call__(self, x):    
        res = 0
        for coeff in self.coefficients:
            res = res * x + coeff
        return res
    
    def __add__(self, other):
        c1 = self.coefficients[::-1]
        c2 = other.coefficients[::-1]
        res = [sum(t) for t in zip_longest(c1, c2, fillvalue=0)]
        return Polynomial(*res[::-1])
    
    def __sub__(self, other):
        c1 = self.coefficients[::-1]
        c2 = other.coefficients[::-1]
        
        res = [t1-t2 for t1, t2 in zip_longest(c1, c2, fillvalue=0)]
        return Polynomial(*res[::-1])
    
    def derivative(self):
        derived_coeffs = []
        exponent = len(self.coefficients) - 1
        for i in range(len(self.coefficients)-1):
            derived_coeffs.append(self.coefficients[i] * exponent)
            exponent -= 1
        return Polynomial(*derived_coeffs)

p1 = Polynomial(1, 0, -4, 3, 0)
p2 = Polynomial(1, 0, -4, 3, 0)
p3 = Polynomial(-0.8, 2.3, 0.5, 1, 0.2)

assert p1 == eval(repr(p1))
assert all((p1(x)==p2(x) for x in range(-10, 10)))

p_sum = p2 + p3
p_diff = p2 - p3
p_der = p2.derivative()

print(p1)
print(p2)
print(p3)
print(p_sum)
print(p_diff)
print(p_der)
