from .baseline import baseline
from .dqn import dqn
from .mcts import mcts


AVAIL_METHODS = {"mcts": mcts, "dqn": dqn, "baseline": baseline}


def get_method(method_name=""):
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
    raise ValueError(
        f"Invalid method name, {method_name}, entered. Must be in {AVAIL_METHODS.keys()}"
    )
