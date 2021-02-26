

import random
import numpy as np


class Actions(object):
    """Common base class for action methods
    """
    def __init__(self, action_space=None, **kwargs):
        if action_space is None:
            raise ValueError('Action space must be defined by action(s) (set)')
        self.action_space = action_space
        self.action_list = self.setup_action_list()

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
        #pass
        raise NotImplementedError()

    def index_to_action(self, action=None):
        """Undefined index to action method:
           Provided an action, return associated index
        """
        #pass
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


class SimpleActions(Actions):
    """SimpleActions for testing purposes
    """
    def __init__(self):
        simple_action_space = ((-30, 1), (-30, 2), (0, 1), (0, 2), (30, 1), (30, 2))
        super().__init__(action_space=simple_action_space)


    #returns action given an index
    def action_to_index(self, a):
        return int(np.trunc(2*(a[0] / 30 + 1) + a[1]))-1


    #returns index of action given an action
    def index_to_action(self, a):
        a = a + 1
        if a % 2 == 0:
            return (int(np.trunc((((a - 2) / 2) - 1) * 30)), 2)
        else:
            return (int(np.trunc((((a - 1) / 2) - 1) * 30)), 1)



