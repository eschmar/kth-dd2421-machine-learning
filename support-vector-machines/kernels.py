import numpy

def linear(x, y):
    """Linear kernel function with constant element 1 term due to elimination of bias."""
    return numpy.dot(x, y) + 1

def polynomial(x, y, p = 2):
    """Polynomial kernel function."""
    return numpy.power(numpy.dot(x, y) + 1, p)

def radialBasis(x, y, sigma):
    """Radial basis kernel function. Adjust sigma to control smoothness."""
    return numpy.exp((numpy.power(numpy.substract(x, y), 2)) / (2 * numpy.power(sigma, 2)))

def sigmoid(x, y, k, delta):
    """Sigmoid kernel function."""
    return numpy.tanh(numpy.dot(k * x, y) - delta)

#  Closures
def polynomialClosure(p = 2):
    """Return the polynomial kernel as a closure."""
    def kernel(x, y):
        return polynomial(x, y, p)

    return kernel

def radialBasisClosure(sigma):
    """Return the radial basis kernel as a closure."""
    def kernel(x, y):
        return radialBasis(x, y, sigma)

    return kernel

def sigmoidClosure(k, delta):
    """Return the sigmoid kernel as a closure."""
    def kernel(x, y):
        return sigmoid(x, y, k, delta)

    return kernel