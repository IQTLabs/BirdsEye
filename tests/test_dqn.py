from birdseye.dqn import dqn
from birdseye.env import RFMultiEnv
from birdseye.actions import WalkingActions
from birdseye.state import RFMultiState
from sigscan import GamutRFSensor


def test_dqn():
    data = {
        'rssi': None,
        'position': None,
        'distance': None,
        'previous_position': None,
        'heading': None,
        'previous_heading': None,
        'course': None,
        'action_proposal': None,
        'action_taken': None,
        'reward': None,
    }
    sensor = GamutRFSensor(
        antenna_filename='radiation_pattern_yagi_5.csv',
        power_tx=str(26),
        directivity_tx=str(1),
        freq=str(5.7e9),
        fading_sigma=str(8),
        threshold=str(-120),
        data=data)

    # Action space
    actions = WalkingActions()
    actions.print_action_info()

    # State managment
    state = RFMultiState(
        n_targets=str(2), reward='heuristic_reward', simulated=True)

    env = RFMultiEnv(sensor=sensor, actions=actions, state=state, simulated=True)
    env.reset()
    dqn(env=env)
