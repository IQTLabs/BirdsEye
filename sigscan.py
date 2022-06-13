import base64
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
from birdseye.utils import get_distance, get_bearing, is_float

data = {} #defaultdict(list)
data.update({
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
})

# Generic data processor 
def data_handler(message_data):
    global data
    data['previous_position'] = data.get('position', None) if not data.get('needs_processing', True) else data.get('previous_position', None)
    data['previous_bearing'] = data.get('bearing', None) if not data.get('needs_processing', True) else data.get('previous_bearing', None)

    data['rssi'] = message_data.get('rssi', None)
    data['position'] = message_data.get('position', None)
    data['course'] = get_bearing(data['previous_position'], data['position'])
    data['bearing'] = -float(message_data.get('bearing', None))+90 if is_float(message_data.get('bearing', None)) else data['course']
    data['distance'] = get_distance(data['previous_position'], data['position'])
    delta_bearing = (data['bearing'] - data['previous_bearing']) if data['bearing'] and data['previous_bearing'] else None
    data['action_taken'] = (delta_bearing, data['distance']) if delta_bearing and data['distance'] else (0,0)

    data['drone_position'] = message_data.get('drone_position', None)
    if data['drone_position']:
        data['drone_position'] = [data['drone_position'][1], data['drone_position'][0]] # swap lon,lat
    
    data['needs_processing'] = True

# MQTT
def on_message(client, userdata, json_message):
    json_data = json.loads(json_message.payload)
    data_handler(json_data)

def on_connect(client, userdata, flags, rc):
    sub_channel = 'gamutrf/rssi'
    print('Connected to {} with result code {}'.format(sub_channel,str(rc)))
    client.subscribe(sub_channel)

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
            print('buffer size = {:.2f} MB'.format(len(buf.getbuffer())/1e6))
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
    reward_func = config.get('reward', 'heuristic_reward')
    n_targets = int(config.get('n_targets', str(2)))
    dqn_checkpoint = config.get('dqn_checkpoint', None)
    if planner_method in ['dqn', 'DQN'] and dqn_checkpoint is None:
        if n_antennas == 1 and antenna_type == 'directional' and n_targets == 2:
            dqn_checkpoint = 'checkpoints/single_directional_entropy_walking.checkpoint'
        elif n_antennas == 1 and antenna_type == 'omni':
            dqn_checkpoint = 'checkpoints/single_omni_entropy_walking.checkpoint'
        elif n_antennas == 2 and antenna_type == 'directional' and n_targets == 2:
            dqn_checkpoint = 'checkpoints/double_directional_entropy_walking.checkpoint'
        elif n_antennas == 2 and antenna_type == 'directional' and n_targets == 1:
            dqn_checkpoint = 'checkpoints/double_directional_entropy_walking_1target.checkpoint'
        elif n_antennas == 1 and antenna_type == 'directional' and n_targets == 1:
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
    global_start_time = datetime.utcnow().timestamp() #datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = birdseye.utils.Results(
        method_name=planner_method,
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
    state = birdseye.state.RFMultiState(n_targets=n_targets, reward=reward_func, simulated=False)

    # Environment
    env = birdseye.env.RFMultiEnv(sensor=sensor, actions=actions, state=state, simulated=False)
    belief = env.reset()

    # Motion planner
    if config.get('use_planner', 'false').lower() != 'true':
        planner = None
    elif planner_method in ['dqn', 'DQN']:
        planner = DQNPlanner(env, actions, device, dqn_checkpoint)
    elif planner_method in ['mcts','MCTS']:
        depth=2
        c=20
        simulations=50
        planner = MCTSPlanner(env, actions, depth, c, simulations)
    else: 
        raise ValueError('planner_method not valid')

    # Flask
    fig = plt.figure(figsize=(18,10), dpi=50)
    ax = fig.subplots()
    fig.set_tight_layout(True)
    time_step = 0
    if config.get('flask', 'false').lower() == 'true':
        run_flask(flask_host, flask_port, fig, results, debug)

    # Main loop
    time.sleep(2)
    while True:
        loop_start = timer()
        data['utc_time'] = datetime.utcnow().timestamp()
        time_step += 1

        if replay_file is not None:
            # load data from saved file
            if time_step-1 == len(replay_ts):
                break
            data_handler(replay_data[replay_ts[time_step-1]])

        action_start = timer()
        data['action_proposal'] = planner.proposal(belief) if planner else [None,None]
        action_end = timer()

        step_start = timer()
        # update belief based on action and sensor observation (sensor is read inside)
        if data.get('needs_processing', False):
            belief, reward, observation = env.real_step(data)
            data['reward'] = reward
            data['needs_processing'] = False
        step_end = timer()

        plot_start = timer()
        results.live_plot(env=env, time_step=time_step, fig=fig, ax=ax, data=data)
        plot_end = timer()

        particle_save_start = timer()
        np.save('{}/{}_particles.npy'.format(results.logdir,data['utc_time']), env.pf.particles)
        particle_save_end = timer()

        data_start = timer()
        with open('{}/birdseye-{}.log'.format(results.logdir, global_start_time), 'a') as outfile:
            json.dump(data, outfile)
            outfile.write('\n')
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

    if config.get('make_gif', 'false').lower() == 'true': 
        results.save_gif('tracking')

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('sigscan_config.ini')
    main(config=config['sigscan'], debug=True)
