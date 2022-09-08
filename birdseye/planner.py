from .dqn import simple_prep
from .dqn import simple_run
from .mcts_utils import MCTSRunner


class PathPlanner:
    def __init__(self, env, actions):
        pass

    def proposal(self, observation):
        pass

class LightweightPlanner(PathPlanner):
    def __init__(self, env, actions):
        self.env = env
        self.actions = actions
    
    def proposal(self, observation):
        # get target estimates
        # select target with minimum std dev
        # calculate void constraint around selected target
        # get possible trajectories (tangent to void and min distance to target)
        ## first element of trajectory turns, then proceed straight
        # for each trajectory calculate void probability functional; if satisfied select action 
        # if none selected, select next best from discrete set
        return None

class MCTSPlanner(PathPlanner):
    def __init__(self, env, actions, depth, c, simulations):
        self.runner = MCTSRunner(env=env, depth=depth, c=c, simulations=simulations)
        self.actions = actions

    def proposal(self, observation):
        return self.runner.run(observation)


class DQNPlanner(PathPlanner):
    def __init__(self, env, actions, device, checkpoint_filename):
        self.model = simple_prep(env, device, checkpoint_filename)
        self.actions = actions
        self.device = device

    def proposal(self, observation):
        return self.actions.index_to_action(
            simple_run(self.model, observation, self.device)
        )
