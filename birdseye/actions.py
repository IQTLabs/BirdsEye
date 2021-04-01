

import random
import itertools
import numpy as np


class Actions(object):
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

    def action_to_index(self, index=0):
        """Undefined action to index method:
           Provided an index, return associated action
        """
        raise NotImplementedError()

    def index_to_action(self, action=None):
        """Undefined index to action method:
           Provided an action, return associated index
        """
        raise NotImplementedError()

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


class SimpleActions(Actions):
    """SimpleActions for testing purposes
    """
    def __init__(self):
        self.del_theta = [-30, 0, 30]
        self.del_r = [1, 2]
        simple_action_space = tuple(itertools.product(self.del_theta, self.del_r))
        super().__init__(action_space=simple_action_space, verbose=True)

    #returns index of action given an action
    def action_to_index(self, action):
        return self.action_space.index(action)

    #returns action given an index
    def index_to_action(self, a_idx):
        return self.action_space[a_idx]


