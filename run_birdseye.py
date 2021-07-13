import sys
import argparse
import configparser
from birdseye.actions import get_action, AVAIL_ACTIONS
from birdseye.sensor import get_sensor, AVAIL_SENSORS
from birdseye.state import get_state, AVAIL_STATES
from birdseye.env import RFEnv
from birdseye.method_utils import get_method, AVAIL_METHODS

def batch_run():


    # Setup requested method objects
    for method_name in ['mcts']:
        for sensor_config in ['drone']:
            for reward in ['range_reward', 'entropy_collision_reward' ]:
                for target_start in ['78']:
                    for target_speed in ['1']:
                        print('===========================')
                        print('Batch Run: ')
                        print('method = {}'.format(method_name))
                        print('sensor_config = {}'.format(sensor_config))
                        print('reward = {}'.format(reward))
                        print('target_speed = {}'.format(target_speed))
                        print('target_start = {}'.format(target_start))
                        print('===========================')
                        run_method = get_method(method_name)
                        action_config = 'simpleactions'
                        state_config = 'rfstate'
                        actions = get_action(action_config)
                        sensor = get_sensor(sensor_config)
                        state = get_state(state_config)

                        # Setup environment
                        env = RFEnv(sensor(), actions(), state(target_speed=target_speed, target_movement=None, target_start=target_start, reward=reward))

                        config = configparser.ConfigParser()
                        config.read(['configs/{}.yaml'.format(method_name)])
                        config.set('Methods', 'action', action_config)
                        config.set('Methods', 'sensor', sensor_config)
                        config.set('Methods', 'state', state_config)
                        config.set('Methods', 'target_speed', target_speed)
                        config.set('Methods', 'target_start', target_start)
                        config.set('Methods', 'reward', reward)

                        # Run the requested algorithm
                        run_method(args=config, env=env)

def run_birdseye(args=None, env=None):
    # Grab Methods information from config file
    config = configparser.ConfigParser()
    config.read([args.config])
    config_dict = dict(config.items('Methods'))
    method_name = config_dict['method']
    action_name = config_dict['action']
    sensor_name = config_dict['sensor']
    state_name = config_dict['state']
    target_speed = config_dict.get('target_speed')
    target_speed_range = config_dict.get('target_speed_range')
    target_movement = config_dict.get('target_movement')
    target_start = config_dict.get('target_start')
    reward = config_dict.get('reward')
    print({section: dict(config[section]) for section in config.sections()})

    # Setup requested method objects
    run_method = get_method(method_name)
    actions = get_action(action_name)
    sensor = get_sensor(sensor_name)
    state = get_state(state_name)

    # Setup environment
    env = RFEnv(sensor(), actions(), state(target_speed=target_speed, target_speed_range=target_speed_range, target_movement=target_movement, target_start=target_start, reward=reward))

    # Run the requested algorithm
    run_method(args=config, env=env)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-b', '--batch',
                             action='store_true',
                             help='Specify batch run option')

    args,remaining_args = arg_parser.parse_known_args()

    if not args.batch:
        arg_parser.add_argument('-c', '--config',
                             help='Specify a configuration file',
                             required=True,
                             metavar='FILE')
        args, remaining_args = arg_parser.parse_known_args(remaining_args, namespace=args)

    if args.batch:
        batch_run()
    else:
        run_birdseye(args=args)
