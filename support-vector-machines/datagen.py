# Matrix operations and helpers
import numpy, pylab, random, math

def generateNormalDistribution(a, b, c, d, e, r):
    return [(random.normalvariate(a, b), random.normalvariate(c, d), e) for i in range(r)]

def generateRandomData(seed = None):
    """Generates random test data."""
    if seed is not None:
        # numpy.random.seed(seed)
        random.seed(100)

    classA = generateNormalDistribution(-1.5, 1, 0.5, 1, 1.0, 5) + generateNormalDistribution(1.5, 1, 0.5, 1, 1.0, 5)
    classB = generateNormalDistribution(0.5, 0.5, -0.5, 0.5, -1.0, 10)
    return classA, classB
