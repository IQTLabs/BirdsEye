import random
import itertools


class Actions:
    """Common base class for action methods

        Parameters
        ----------
        action_space : tuple
            Set of tuples defining combinations of actions
            for all dimensions
    """
    def __init__(self, action_space=None, verbose=False, **kwargs):
        if action_space is None:
            raise ValueError('Action space must be defined by action(s) (set)')
        self.action_space = action_space
        self.action_list = self.setup_action_list()
        self.verbose = verbose

        if verbose:
            self.print_action_info()

    def avail_actions(self):
        """Return set of available actions
        """
        return self.action_space

    def get_action_list(self):
        """Return ordered list of actions
        """
        return self.action_list

    #returns index of action given an action
    def action_to_index(self, action):
        return self.action_space.index(action)

    #returns action given an index
    def index_to_action(self, a_idx):
        return self.action_space[a_idx]

    def setup_action_list(self):
        """Define ordered list of actions
        """
        return list(map(self.action_to_index, self.action_space))

    def get_random_action(self):
        """Return random action and associated index
        """
        random_action_index = random.choice(self.get_action_list())
        return self.index_to_action(random_action_index), random_action_index

    def print_action_info(self):
        print("Available Actions:")
        print("  ID, Values")
        for ai in zip(self.get_action_list(), self.avail_actions()):
            print("   {}   {}".format(ai[0], ai[1]))

# Human walking action space
class WalkingActions(Actions):
    """WalkingActions for a human walking
    """
    def __init__(self):
        # change in heading
        self.del_theta = [-45, 0, 45]
        # speed
        self.del_r = [0,1.5]
        simple_action_space = tuple(itertools.product(self.del_theta, self.del_r))
        super().__init__(action_space=simple_action_space, verbose=False)

class SimpleActions(Actions):
    """SimpleActions for testing purposes
    """
    def __init__(self):
        self.del_theta = [-30, 0, 30]
        self.del_r = [0,4]
        simple_action_space = tuple(itertools.product(self.del_theta, self.del_r))
        super().__init__(action_space=simple_action_space, verbose=False)

    #returns index of action given an action
    def action_to_index(self, action):
        return self.action_space.index(action)

    #returns action given an index
    def index_to_action(self, a_idx):
        return self.action_space[a_idx]

class BaselineActions(Actions):
    """SimpleActions for testing purposes
    """
    def __init__(self):
        self.del_theta = [-30, 0, 30]
        self.del_r = [0,4]
        baseline_action_space = tuple(itertools.product(self.del_theta, self.del_r))
        super().__init__(action_space=baseline_action_space, verbose=False)

    #returns index of action given an action
    def action_to_index(self, action):
        return self.action_space.index(action)

    #returns action given an index
    def index_to_action(self, a_idx):
        return self.action_space[a_idx]


AVAIL_ACTIONS = {'simpleactions' : SimpleActions,
                 'baselineactions': BaselineActions,
                 'walkingactions': WalkingActions
                }

def get_action(action_name=''):
    """Convenience function for retrieving BirdsEye action methods
    Parameters
    ----------
    action_name : {'simpleactions'}
        Name of action method.
    Returns
    -------
    action_obj : Action class object
        BirdsEye action method.
    """
    action_name = action_name.lower()
    if action_name in AVAIL_ACTIONS:
        action_obj = AVAIL_ACTIONS[action_name]
        return action_obj
    else:
        raise ValueError('Invalid action method name, {}, entered. Must be '
                         'in {}'.format(action_name, AVAIL_ACTIONS.keys()))

