import base64
import itertools
import json
import threading
import time
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from timeit import default_timer as timer

import configparser
import paho.mqtt.client as mqtt
import torch
import numpy as np
from flask import Flask
import matplotlib.pyplot as plt

import birdseye.sensor
import birdseye.env
import birdseye.state
import birdseye.utils
import birdseye.dqn
import birdseye.mcts_utils
from birdseye.actions import WalkingActions
from birdseye.planner import MCTSPlanner, DQNPlanner
from birdseye.utils import get_distance, get_bearing

data = defaultdict(list)
data.update({
    'rssi': None,
    'position': None,
    'distance': None,
    'previous_position': None,
    'bearing': 0,
    'previous_bearing': 0,
    'action_proposal': [],
    'action_taken': [],
    'reward': [],
})

# MQTT
def on_message(client, userdata, message):
    global data
    json_message = json.loads(message.payload)
    data['rssi'] = json_message['rssi']
    data['position'] = json_message['position']
    if data.get('previous_position', None) is not None:
        data['bearing'] = get_bearing(data['previous_position'], data['position'])
        data['distance'] = get_distance(data['previous_position'], data['position'])
        if data.get('previous_bearing', None) is not None:
            delta_bearing = data['bearing'] - data['previous_bearing']
            data['action'] = (delta_bearing, data['distance'])
        data['previous_bearing'] = data['bearing']
    data['previous_position'] = json_message['position']
    #print('data: ',data)

def on_connect(client, userdata, flags, rc):
    sub_channel = 'gamutrf/rssi'
    print('Connected to {} with result code {}'.format(sub_channel,str(rc)))
    client.subscribe(sub_channel)

# Replay data from file 
def replay_handler(replay_data):
    global data
    data['rssi'] = replay_data.get('rssi', None)
    data['position'] = replay_data.get('position', None)
    data['drone_position'] = replay_data.get('drone_position', None)
    
    if data.get('position', None) is not None:
        if data.get('previous_position', None) is not None:
            data['bearing'] = get_bearing(data['previous_position'], data['position'])
            data['distance'] = get_distance(data['previous_position'], data['position'])
            if data.get('previous_bearing', None) is not None:
                delta_bearing = data['bearing'] - data['previous_bearing']
                data['action'] = (delta_bearing, data['distance'])
            data['previous_bearing'] = data['bearing']
        data['previous_position'] = data['position']
    else:
        data['action'] = (0,0)

# GamutRF Sensor
class GamutRFSensor(birdseye.sensor.SingleRSSI):
    def __init__(self, antenna_filename=None, power_tx=26, directivity_tx=1, f=5.7e9, fading_sigma=None, threshold=-120):
        super().__init__(antenna_filename=antenna_filename, power_tx=power_tx, directivity_tx=directivity_tx, f=f, fading_sigma=fading_sigma)
        self.threshold = threshold

    def real_observation(self):
        global data
        if (data.get('rssi', None)) is None or (data['rssi'] < self.threshold):
            return None
            #return -40
        return data['rssi']

# Flask
def run_flask(flask_host, flask_port, fig, results, debug):
    app = Flask(__name__)

    @app.route("/")
    def hello():
        # Save figure to a temporary buffer.
        flask_start_time = timer()
        buf = BytesIO()

        try:
            fig.savefig(buf, format='png', bbox_inches='tight')
        except ValueError:
            return '<html><head><meta http-equiv="refresh" content="1"></head><body><p>No image, refreshing...</p></body></html>'

        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        flask_end_time = timer()

        if debug:
            print('=======================================')
            print('Flask Timing')
            print('time step = ',results.time_step)
            print('buffer size = {:.2f} MB'.format(buf.getbuffer().nbytes/1e6))
            print('Duration = {:.4f} s'.format(flask_end_time - flask_start_time))
            print('=======================================')
        return f'<html><head><meta http-equiv="refresh" content="0.5"></head><body><img src="data:image/png;base64,{data}"/></body></html>'

    host_name = flask_host #'0.0.0.0'
    port = flask_port #4999
    threading.Thread(target=lambda: app.run(host=host_name, port=port, debug=True, use_reloader=False)).start()


# main loop
def main(config=None, debug=False):
    global data

    replay_file = config.get('replay_file', None)

    mqtt_host = config.get('mqtt_host', 'localhost')
    mqtt_port = int(config.get('mqtt_port', str(1883)))

    flask_host = config.get('flask_host', '0.0.0.0')
    flask_port = int(config.get('flask_port', str(4999)))

    n_antennas = int(config.get('n_antennas', str(1)))
    antenna_type = config.get('antenna_type', 'omni')
    planner_method = config.get('planner_method', 'dqn')
    power_tx = float(config.get('power_tx', str(26)))
    directivity_tx = float(config.get('directivity_tx', str(1)))
    freq = float(config.get('f', str(5.7e9)))
    fading_sigma = float(config.get('fading_sigma', str(8)))
    threshold = float(config.get('threshold', str(-120)))
    reward = config.get('reward', 'heuristic_reward')
    n_targets = int(config.get('n_targets', str(2)))
    dqn_checkpoint = config.get('dqn_checkpoint', None)
    if planner_method in ['dqn', 'DQN'] and dqn_checkpoint is None:
        if n_antennas == 1 and antenna_type == 'directional':
            dqn_checkpoint = 'checkpoints/single_directional_entropy_walking.checkpoint'
        elif n_antennas == 1 and antenna_type == 'omni':
            dqn_checkpoint = 'checkpoints/single_omni_entropy_walking.checkpoint'
        elif n_antennas == 2 and antenna_type == 'directional' and n_targets == 2:
            dqn_checkpoint = 'checkpoints/double_directional_entropy_walking.checkpoint'
        elif n_antennas == 2 and antenna_type == 'directional' and n_targets == 1:
            dqn_checkpoint = 'checkpoints/double_directional_entropy_walking_1target.checkpoint'

    # MQTT
    if replay_file is None:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(mqtt_host, mqtt_port, 60)
        client.loop_start()
    else:
        with open(replay_file, 'r') as open_file:
            replay_data = json.load(open_file)
            replay_ts = sorted(replay_data.keys())

    # BirdsEye
    global_start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = birdseye.utils.Results(method_name=planner_method,
                        global_start_time=global_start_time,
                        config=config)

    # Sensor
    if antenna_type in ['directional', 'yagi', 'logp']:
        antenna_filename = 'radiation_pattern_yagi_5.csv'
    elif antenna_type in ['omni', 'omnidirectional']:
        antenna_filename = 'radiation_pattern_monopole.csv'

    sensor = GamutRFSensor(
        antenna_filename=antenna_filename,
        power_tx=power_tx,
        directivity_tx=directivity_tx,
        f=freq,
        fading_sigma=fading_sigma,
        threshold=threshold) # fading sigm = 8dB, threshold = -120dB

    # Action space
    actions = WalkingActions()
    actions.print_action_info()

    # State managment
    state = birdseye.state.RFMultiState(n_targets=n_targets, reward=reward, simulated=False)

    # Environment
    env = birdseye.env.RFMultiEnv(sensor=sensor, actions=actions, state=state, simulated=False)
    belief = env.reset()

    # Motion planner
    if config.get('use_planner', 'false').lower() != 'true':
        planner = None
    elif planner_method in ['dqn', 'DQN']:
        planner = DQNPlanner(env, actions, device, dqn_checkpoint)
    elif planner_method in ['mcts','MCTS']:
        depth=10
        c=20
        simulations=100
        planner = MCTSPlanner(env, actions, depth, c, simulations)
    else: 
        raise ValueError('planner_method not valid')

    # Flask
    fig = plt.figure(figsize=(10,10))
    ax = fig.subplots()
    time_step = 0
    if config.get('flask', 'false').lower() == 'true':
        run_flask(flask_host, flask_port, fig, results, debug)

    # Main loop
    time.sleep(2)
    while True:
        loop_start = timer()
        time_step += 1

        if replay_file is not None:
            # load data from saved file
            replay_handler(replay_data[replay_ts[time_step-1]])

        action_start = timer()
        action_proposal = planner.proposal(belief) if planner else [None,None]
        action_taken = data.get('action', (0,0))
        action_end = timer()

        step_start = timer()
        # update belief based on action and sensor observation (sensor is read inside)
        belief, reward, observation = env.real_step(action_taken, data.get('bearing', None))
        step_end = timer()

        #textstr = ['Actual\nBearing = {:.0f} deg\nSpeed = {:.2f} m/s'.format(data.get('bearing', 0),data.get('distance', 0)), 'Proposed\nBearing = {:.0f} deg\nSpeed = {:.2f} m/s'.format(data.get('bearing',0)+action_proposal[0],action_proposal[1])]
        textstr = ['Actual\nBearing = {:.0f} deg\nSpeed = {:.2f} m/s'.format(data.get('bearing', 0),action_taken[1]), 'Proposed\nBearing = {} deg\nSpeed = {} m/s'.format(action_proposal[0],action_proposal[1])]

        plot_start = timer()
        results.live_plot(env=env, time_step=time_step, fig=fig, ax=ax, data=data, simulated=False, textstr=textstr)
        if config.get('native_plot', 'false').lower() == 'true':
            plt.draw()
            plt.pause(0.001)
        plot_end = timer()

        particle_save_start = timer()
        np.save('{}/{}_particles.npy'.format(results.logdir,int(time.time())), env.pf.particles)
        particle_save_end = timer()

        data_start = timer()
        data['action_proposal'].append(action_proposal)
        data['action_taken'].append(action_taken)
        data['reward'].append(reward)
        data['observation'].append(observation)
        for key in data:
            np.save('{}/{}.npy'.format(results.logdir, key), data[key])
        data_end = timer()

        loop_end = timer()

        if debug:
            print('=======================================')
            print('BirdsEye Timing')
            print('time step = {}'.format(time_step))
            print('action selection = {:.4f} s'.format(action_end-action_start))
            print('env step = {:.4f} s'.format(step_end - step_start))
            print('plot = {:.4f} s'.format(plot_end-plot_start))
            print('particle save = {:.4f} s'.format(particle_save_end - particle_save_start))
            print('data save = {:.4f} s'.format(data_end-data_start))
            print('main loop = {:.4f} s'.format(loop_end-loop_start))
            print('=======================================')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('sigscan_config.ini')
    main(config=config['sigscan'], debug=True)
