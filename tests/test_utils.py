import matplotlib.pyplot as plt

from birdseye.env import RFMultiEnv
from birdseye.actions import WalkingActions
from birdseye.state import RFMultiState
from birdseye.utils import GPSVis
from birdseye.utils import Results
from sigscan import GamutRFSensor


def test_gpsvis():
    # TODO fix type error
    """
>           np.linspace(self.bounds[1], self.bounds[3], num=8))
E       TypeError: 'NoneType' object is not subscriptable
    """
    instance = GPSVis(position=[0,0], bounds=[0,1,2,3])
    instance.plot_map(output='save')
    instance.plot_map(output='plot')


def test_results():
    instance = Results(plotting='True')
    instance = Results(plotting='false')

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
    fig, axis1 = plt.subplots(figsize=(10, 13))
    instance.build_multitarget_plots(env=env, fig=fig)
    instance.build_plots(belief=env.pf.particles)
