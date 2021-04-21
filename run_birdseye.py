import sys
import argparse
import configparser
from birdseye.actions import get_action
from birdseye.sensor import get_sensor
from birdseye.state import get_state
from birdseye.env import RFEnv
from birdseye.method_utils import get_method


def run_birdseye(args=None, env=None):
    # Configuration file parser
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('-c', '--config',
                             help='Specify a configuration file',
                             required=True,
                             metavar='FILE')
    if args is not None: 
        args = conf_parser.parse_known_args(args)[0]
    else: 
        args = conf_parser.parse_known_args()[0]

    # Grab Methods information from config file
    config = configparser.ConfigParser()
    config.read([args.config])
    config_dict = dict(config.items('Methods'))
    method_name = config_dict['method']
    action_name = config_dict['action']
    sensor_name = config_dict['sensor']
    state_name = config_dict['state']
    target_speed = config_dict.get('target_speed') 
    target_movement = config_dict.get('target_movement')
    print({section: dict(config[section]) for section in config.sections()})

    # Setup requested method objects
    run_method = get_method(method_name)
    actions = get_action(action_name)
    sensor = get_sensor(sensor_name)
    state = get_state(state_name)
    
    # Setup environment
    env = RFEnv(sensor(), actions(), state(target_speed=target_speed, target_movement=target_movement))

    # Run the requested algorithm
    run_method(args=config, env=env)


if __name__ == '__main__':
    run_birdseye(args=sys.argv[1:])
