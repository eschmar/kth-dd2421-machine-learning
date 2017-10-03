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