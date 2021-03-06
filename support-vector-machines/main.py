import sys, helper

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
def run(args):
    params = helper.parseArguments(args)

    # Generate base data
    classA, classB = datagen.generateRandomData(100)
    data = classA + classB
    random.shuffle(data)
    n = len(data)

    p = 3
    if '-p' in params:
        p = int(params['-p'])

    sigma = 2
    if '-sigma' in params:
        sigma = float(params['-sigma'])

    k = 0.05
    if '-k' in params:
        k = float(params['-k'])

    delta = 0
    if '-delta' in params:
        delta = int(params['-delta'])

    C = 0
    if '-C' in params:
        C = float(params['-C'])

    # Choose kernel
    if '--polynomial' in params:
        kernel = kernels.polynomialClosure(p)
    elif '--radial' in params:
        kernel = kernels.radialBasisClosure(sigma)
    elif '--sigmoid' in params:
        kernel = kernels.sigmoidClosure(k, delta)
    else:
        kernel = kernels.linear

    # Create dual formulation terms
    # P_i,j = t_i * t_j * K(\vec{x_i}, \vec{x_j})
    P = []
    for i in range(n):
        P.append([])

    for i in range(n):
        for j in range(n):
            P[i].append(data[i][2] * data[j][2] * kernel([data[i][0], data[i][1]], [data[j][0], data[j][1]]))

    q = -1 * numpy.ones((n, 1))
    h = numpy.zeros_like(q)
    G = -1 * numpy.eye(n)

    # Slack variables
    if C != 0:
        G = numpy.concatenate((numpy.eye(n), G))
        h = numpy.concatenate((C * numpy.ones((n, 1)), h))

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
    pylab.clf()
    pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
    pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    if '--save' in params:
        filename = 'foo.png'
        if '-o' in params:
            filename = params['-o']
        # pylab.savefig("out/{0}".format(filename), bbox_inches='tight')
        pylab.savefig("out/{0}".format(filename))
    else:
        pylab.show();

if __name__ == '__main__':
    run(sys.argv[1:])
