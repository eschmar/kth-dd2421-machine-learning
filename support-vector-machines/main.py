# Convex optimization packages
from cvxopt.solvers import qp
from cvxopt.base import matrix

# Matrix operations and helpers
import numpy, pylab, random, math

# Own
import kernels, datagen

# Helpers
def extractSupportVectors(data, alpha, threshold = 1.e-5):
    """Extracts support vectors in format [x, y, t, a]."""
    vectors = []
    for i in range(len(alpha)):
        if alpha[i] > threshold:
            vectors.append([data[i][0], data[i][1], data[i][2], alpha[i]])

    return vectors

def generateIndiciator(data, alpha, kernel):
    """Returns the indicator function as a closure(x, y)."""
    vectors = extractSupportVectors(data, alpha)
    def indicator(x, y):
        result = 0
        for i in range(len(vectors)):
            result += vectors[i][3] * vectors[i][2] * kernel([x, y], [vectors[i][0], vectors[i][1]])
        return result

    return indicator

# Run
if __name__ == '__main__':
    # Generate base data
    classA, classB = datagen.generateRandomData(100)
    data = classA + classB
    random.shuffle(data)
    n = len(data)

    # Choose kernel
    # kernel = kernels.linear
    # kernel = kernels.polynomialClosure(3)
    kernel = kernels.radialBasisClosure(0.35)
    # kernel = kernels.sigmoidClosure(0.2, 0.5)

    # Create dual formulation terms
    # P_i,j = t_i * t_j * K(\vec{x_i}, \vec{x_j})
    P = []
    for i in range(n):
        P.append([])

    for i in range(n):
        for j in range(n):
            P[i].append(data[i][2] * data[j][2] * kernel([data[i][0], data[i][1]], [data[j][0], data[j][1]]))

    q = -1 * numpy.ones(n)
    h = numpy.zeros_like(q)
    G = -1 * numpy.eye(n)

    # Find \vec{a} which minimizes dual formulation
    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alpha = list(r['x'])

    # Set up indicator closure
    indicator = generateIndiciator(data, alpha, kernel)

    # Find decision boundary
    xrange = numpy.arange(-4, 4, 0.05)
    yrange = numpy.arange(-4, 4, 0.05)
    grid = matrix([[indicator(x, y) for y in yrange] for x in xrange])

    #  plot all the things
    # pylab.hold(True)
    pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
    pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    pylab.show();
