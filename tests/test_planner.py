import torch

import birdseye.env
from birdseye.actions import WalkingActions
from birdseye.planner import PathPlanner
from birdseye.planner import DQNPlanner


def test_path_planner():
    planner = PathPlanner(env=None, config=None, device=None)
    planner.proposal(None)


def test_dqn_planner():
    actions = WalkingActions()
    actions.print_action_info()
    device = torch.device('cpu')
    dqn_checkpoint = 'checkpoints/single_directional_entropy_walking_1target.checkpoint'
    env = birdseye.env.RFMultiEnv(
            sensor=sensor, actions=actions, state=state, simulated=False)
    belief = env.reset()
    planner = DQNPlanner(env, actions, device, dqn_checkpoint)
    proposal = planner.proposal(belief)
