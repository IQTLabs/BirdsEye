
from .mcts import mcts
from .dqn import dqn
from .baseline import baseline


AVAIL_METHODS = {'mcts' : mcts,
                 'dqn' : dqn,
                 'baseline' : baseline}

def get_method(method_name=''):
    """Convenience function for retrieving BirdsEye methods
    Parameters
    ----------
    method_name : {'mcts', 'dqn'}
        Name of method.
    Returns
    -------
    method : function
        BirdsEye method.
    """
    method_name = method_name.lower()
    if method_name in AVAIL_METHODS:
        method = AVAIL_METHODS[method_name]
        return method
    else:
        raise ValueError('Invalid method name, {}, entered. Must be '
                         'in {}'.format(method_name, AVAIL_METHODS.keys()))


