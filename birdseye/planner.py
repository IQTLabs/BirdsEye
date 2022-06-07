
from .dqn import simple_prep, simple_run
from .mcts_utils import MCTSRunner


class PathPlanner():

    def __init__(self, env, config, device):
        pass

    def proposal(self, observation):
        pass


class MCTSPlanner(PathPlanner):

    def __init__(self, env, actions, depth, c, simulations):
        self.runner = MCTSRunner(env=env, depth=depth, c=c, simulations=simulations)
        self.actions = actions

    def proposal(self, observation):
        return self.actions.action_to_index(self.runner.run(observation))


class DQNPlanner(PathPlanner):

    def __init__(self, env, device, checkpoint_filename):
        self.model = simple_prep(env, device, checkpoint_filename)
        self.device = device

    def proposal(self, observation):
        return simple_run(self.model, observation, self.device)
