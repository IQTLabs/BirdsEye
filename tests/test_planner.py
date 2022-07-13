"""
Tests for planner.py
"""
import torch

import birdseye.env

from birdseye.actions import WalkingActions
from birdseye.planner import PathPlanner
from birdseye.planner import DQNPlanner
from birdseye.state import RFMultiState
from sigscan import GamutRFSensor


def test_path_planner():
    """
    Test the PathPlanner class
    """
    planner = PathPlanner(env=None, config=None, device=None)
    planner.proposal(None)


def test_dqn_planner():
    """
    Test the DQNPlanner class
    """
    actions = WalkingActions()
    actions.print_action_info()
    device = torch.device("cpu")
    dqn_checkpoint = "checkpoints/single_directional_entropy_walking_1target.checkpoint"
    data = {
        "rssi": None,
        "position": None,
        "distance": None,
        "previous_position": None,
        "heading": None,
        "previous_heading": None,
        "course": None,
        "action_proposal": None,
        "action_taken": None,
        "reward": None,
    }
    sensor = GamutRFSensor(
        antenna_filename="radiation_pattern_yagi_5.csv",
        power_tx=str(26),
        directivity_tx=str(1),
        freq=str(5.7e9),
        fading_sigma=str(8),
        threshold=str(-120),
        data=data,
    )
    state = RFMultiState(n_targets=str(1), reward="heuristic_reward", simulated=True)
    env = birdseye.env.RFMultiEnv(
        sensor=sensor, actions=actions, state=state, simulated=False
    )
    belief = env.reset()
    planner = DQNPlanner(env, actions, device, dqn_checkpoint)
    proposal = planner.proposal(belief)
