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
    # TODO fix attribute error
    """
    def build_plots(self, xp=[], belief=[], abs_sensor=None, abs_target=None, abs_particles=None, time_step=None, fig=None, ax=None):
>       print(belief.shape)
E       AttributeError: 'list' object has no attribute 'shape'
    """
    #instance.build_plots()

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
    env.reset()
    fig, axis1 = plt.subplots(figsize=(10, 13))
    instance.build_multitarget_plots(env=env, fig=fig)
