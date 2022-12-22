from collections import namedtuple

ModelInput = namedtuple('ModelInput', ('x', 'y', 'index'))
ModelOutput = namedtuple('ModelOutput', ('y_prime', 'score', 'var'), defaults=(None, None, None))
