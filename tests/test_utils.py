from birdseye.env import RFMultiEnv
from birdseye.actions import WalkingActions
from birdseye.state import RFMultiState
from birdseye.utils import GPSVis
from birdseye.utils import Results


def test_gpsvis():
    instance = GPSVis()
    instance.plot_map(output='save')
    instance.plot_map(output='plot')


def test_results():
    instance = Results(plotting='True')
    instance = Results(plotting='false')
    instance.build_plots()

    data = {
        'rssi': None,
        'position': None,
        'distance': None,
        'previous_position': None,
        'bearing': None,
        'previous_bearing': None,
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
    instance.build_multitarget_plots(env=env)
