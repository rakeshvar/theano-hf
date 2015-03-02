# Author: Nicolas Boulanger-Lewandowski
# University of Montreal, 2012-2013


import sys
import numpy
import theano
import theano.tensor as tt

from hf import hf_optimizer, SequenceDataset


def test_cg(n=500):
    '''Attempt to solve a linear system using the CG function in 
    hf_optimizer.'''

    A = numpy.random.uniform(-1, 1, (n, n))
    A = numpy.dot(A.T, A)
    val, vec = numpy.linalg.eig(A)
    val = numpy.random.uniform(1, 5000, (n, 1))
    A = numpy.dot(vec.T, val*vec)
    
    # hack into a fake hf_optimizer object
    x = theano.shared(0.0)
    s = 2.0*x
    hf = hf_optimizer([x], [], s, [s**2])
    hf.quick_cost = lambda *args, **kwargs: 0.0
    hf.global_backtracking = False
    hf.preconditioner = False
    hf.max_cg_iterations = 300
    hf.batch_Gv = lambda v: numpy.dot(A, v)
    b = numpy.random.random(n)
    c, x, j, i = hf.cg(b)
    print()
    
    print( 'error on b =', abs(numpy.dot(A, x) - b).mean())
    print( 'error on x =', abs(numpy.linalg.solve(A, b) - x).mean())


def sgd_optimizer(p, inputs, costs, train_set, lr=1e-4):
  '''SGD optimizer with a similar interface to hf_optimizer.'''

  g = [tt.grad(costs[0], i) for i in p]
  updates = dict((i, i - lr*j) for i, j in zip(p, g))
  f = theano.function(inputs, costs, updates=updates)
  
  try:
    for u in range(1000):
      cost = []
      for i in train_set.iterate(True):
        cost.append(f(*i))
      print( 'update %i, cost=' %u, numpy.mean(cost, axis=0))
      sys.stdout.flush()

  except KeyboardInterrupt: 
    print( 'Training interrupted.')


# feed-forward neural network with sigmoidal output
def simple_NN(sizes=(784, 100, 10)):
    x = tt.matrix()
    t = tt.matrix()

    p = []
    y = x

    for i in range(len(sizes)-1):
        a, b = sizes[i:i+2]
        Wi = theano.shared((10./numpy.sqrt(a+b)
          * numpy.random.uniform(-1, 1, size=(a, b))).astype(theano.config.floatX))
        bi = theano.shared(numpy.zeros(b, dtype=theano.config.floatX))
        p += [Wi, bi]

        s = tt.dot(y,Wi) + bi
        y = tt.nnet.sigmoid(s)

    c = (-t* tt.log(y) - (1-t)* tt.log(1-y)).mean()
    acc = tt.neq(tt.round(y), t).mean()

    return p, [x, t], s, [c, acc]


def example_NN(hf=True):
    p, inputs, s, costs = simple_NN((2, 50, 40, 30, 1))

    xor_dataset = [[], []]
    for i in range(50000):
        x = numpy.random.randint(0, 2, (50, 2))
        t = (x[:, 0:1] ^ x[:, 1:2]).astype(theano.config.floatX)
        x = x.astype(theano.config.floatX)
        xor_dataset[0].append(x)
        xor_dataset[1].append(t)

    training_examples = len(xor_dataset[0]) * 3/4
    train = [xor_dataset[0][:training_examples], xor_dataset[1][:training_examples]]
    valid = [xor_dataset[0][training_examples:], xor_dataset[1][training_examples:]]

    gradient_dataset = SequenceDataset(train, batch_size=None, nbatches=10000)
    cg_dataset = SequenceDataset(train, batch_size=None, nbatches=5000)
    valid_dataset = SequenceDataset(valid, batch_size=None, nbatches=5000)

    if hf:
        hf_optimizer(p, inputs, s, costs).train(
            gradient_dataset,
            cg_dataset,
            initial_lambda=1.0,
            preconditioner=True,
            validation=valid_dataset)
    else:
        sgd_optimizer(p, inputs, costs, gradient_dataset, lr=1e-3)
    

# single-layer recurrent neural network with sigmoid output, only last time-step output is significant
def runif(shape):
    return theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, shape).astype(
        theano.config.floatX))


def simple_RNN(nh):
    Wx = runif((1, nh))
    Wh = runif((nh, nh))
    Wy = runif((nh, 1))
    bh = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
    by = theano.shared(numpy.zeros(1, dtype=theano.config.floatX))
    h0 = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
    p = [Wx, Wh, Wy, bh, by, h0]

    x = tt.matrix()

    def recurrence(x_t, h_tm1):
        ha_t = tt.dot(x_t, Wx) + tt.dot(h_tm1, Wh) + bh
        h_t = tt.tanh(ha_t)
        s_t = tt.dot(h_t, Wy) + by
        return [ha_t, h_t, s_t]

    ([ha, h, activations], updates) = theano.scan(fn=recurrence,
                                              sequences=x,
                                              outputs_info=[dict(), h0, dict()])

    h = tt.tanh(ha)  # so it is differentiable with respect to ha
    t = x[0, 0]
    s = activations[-1, 0]
    y = tt.nnet.sigmoid(s)
    loss = -t*tt.log(y + 1e-14) - (1-t)*tt.log((1-y) + 1e-14)
    acc = tt.neq(tt.round(y), t)

    return p, [x], s, [loss, acc], h, ha


def example_RNN(hf=True):
    p, inputs, s, costs, h, ha = simple_RNN(100)

    memorization_dataset = [[]]  # memorize the first unit for 100 time-steps with binary noise
    for i in range(100000):
        memorization_dataset[0].append(
            numpy.random.randint(2, size=(100, 1)).astype(theano.config.floatX))

    train = [memorization_dataset[0][:-1000]]
    valid = [memorization_dataset[0][-1000:]]

    gradient_dataset = SequenceDataset(train, batch_size=None, nbatches=5000)
    cg_dataset = SequenceDataset(train, batch_size=None, nbatches=1000)
    valid_dataset = SequenceDataset(valid, batch_size=None, nbatches=1000)

    if hf:
        hf_optimizer(p, inputs, s, costs, 0.5*(h + 1), ha).train(
            gradient_dataset,
            cg_dataset,
            initial_lambda=0.5,
            mu=1.0,
            preconditioner=False,
            validation=valid_dataset)
    else:
        sgd_optimizer(p, inputs, costs, gradient_dataset, lr=5e-5)
