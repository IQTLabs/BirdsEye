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
    dqn_defaults = {
        'number_timesteps': 200,
        'dueling': False,
        'double_q': False,
        'param_noise': True,
        'exploration_fraction': 0.2,
        'exploration_final_eps': 0.1,
        'batch_size': 100,
        'train_freq': 4,
        'learning_starts': 100,
        'target_network_update_freq': 100,
        'buffer_size': 10000,
        'prioritized_replay': True,
        'prioritized_replay_alpha': 0.6,
        'prioritized_replay_beta0': 0.4,
        'min_value': -10,
        'max_value': 10,
        'max_episode_length': 500,
        'atom_num': 1,
        'ob_scale': 1,
        'gamma': 0.99,
        'grad_norm': 10.0,
        'save_interval': 100,
        'eval_interval': 100,
        'save_path': 'checkpoints',
        'log_path': 'rl_log',
        'use_gpu': True,
        'plotting': False,
        'trials': 1,
        'eval_mode': False
    }
    dqn(env=env, dqn_defaults=dqn_defaults)
