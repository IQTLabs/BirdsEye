import paho.mqtt.client as mqtt
import numpy as np 
from datetime import datetime
import itertools
import torch 
import time
import json
import matplotlib.pyplot as plt
import base64
import random
from io import BytesIO
from flask import Flask
from matplotlib.figure import Figure
import threading
from collections import defaultdict

import birdseye.sensor 
import birdseye.env 
import birdseye.actions
import birdseye.state
import birdseye.utils
import birdseye.dqn 
import birdseye.mcts_utils

data = {}

# helper functions for lat/lon
def get_distance(coord1, coord2): 
    lat1, long1 = coord1
    lat2, long2 = coord2
    # approximate radius of earth in km
    R = 6373.0

    lat1 = np.radians(lat1) 
    long1 = np.radians(long1) 

    lat2 = np.radians(lat2) 
    long2 = np.radians(long2) 

    dlon = long2 - long1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance*(1e3)
    
    
def get_bearing(coord1, coord2):
    lat1, long1 = coord1
    lat2, long2 = coord2
    dLon = (long2 - long1)
    x = np.cos(np.radians(lat2)) * np.sin(np.radians(dLon))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)

    return -brng + 90

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
            print('bearing = ',data['bearing'])
            print('delta_bearing = ',delta_bearing) 
        data['previous_bearing'] = data['bearing']
    data['previous_position'] = json_message['position']
    #print('data: ',data)

def on_connect(client, userdata, flags, rc):
    sub_channel = 'gamutRF'
    print('Connected to {} with result code {}'.format(sub_channel,str(rc)))
    client.subscribe(sub_channel)


# GamutRF Sensor 
class GamutRFSensor(birdseye.sensor.SingleRSSI): 
    def __init__(self, fading_sigma=None, threshold=-120): 
        super().__init__(fading_sigma)
        self.threshold = threshold

    def real_observation(self): 
        global data 
        if (data.get('rssi', None)) is None or (data['rssi'] < self.threshold): 
            return None 
            #return -40
        return data['rssi']

# Human walking action space 
class WalkingActions(birdseye.actions.Actions):
    """WalkingActions for a human walking
    """
    def __init__(self):
        # change in heading 
        self.del_theta = [-45, 0, 45]
        # speed 
        self.del_r = [0,1.5]
        simple_action_space = tuple(itertools.product(self.del_theta, self.del_r))
        super().__init__(action_space=simple_action_space, verbose=False)

# Path Planners 
class PathPlanner(): 
    def __init__(self, env, config, device): 
        pass
    def proposal(self, observation): 
        pass

class MCTSPlanner(PathPlanner): 
    def __init__(self, env, actions, depth, c, simulations):
        self.runner =  birdseye.mcts.MCTSRunner(env=env, depth=depth, c=c, simulations=simulations)
        self.actions = actions 
    def proposal(self, observation): 
        return self.actions.action_to_idx(self.runner.run(observation))

class DQNPlanner(PathPlanner): 
    def __init__(self, env, device): 
        self.model = birdseye.dqn.simple_prep(env, device)
        self.device = device
    def proposal(self, observation): 
        return birdseye.dqn.simple_run(self.model, observation, self.device)
        
# Flask 
def run_flask(fig, results): 
    app = Flask(__name__)

    @app.route("/")
    def hello():
        # Save figure to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        png_filename = '{}/png/{}.png'.format(results.gif_dir, results.time_step)
        #fig.savefig(png_filename, format='png', bbox_inches='tight')
        print('save timestep ',results.time_step)
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f'<html><head><meta http-equiv="refresh" content="0.5"></head><body><img src="data:image/png;base64,{data}"/></body></html>'

    host_name = '0.0.0.0'
    port = 4999
    threading.Thread(target=lambda: app.run(host=host_name, port=port, debug=True, use_reloader=False)).start()
    
# main loop 
def main(config=None): 

    # MQTT 
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message 
    client.connect('ljt.dynamic.ucsd.edu', 1883, 60) 
    client.loop_start()

    # BirdsEye 
    global_start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = birdseye.utils.Results(method_name='dqn',
                        global_start_time=global_start_time,
                        plotting=True,
                        config=config)
    sensor = GamutRFSensor(fading_sigma=8, threshold=-120) # fading sigm = 8dB, threshold = -120dB
    actions = WalkingActions()
    actions.print_action_info()
    state = birdseye.state.RFMultiState(n_targets=2, reward='entropy_collision_reward', simulated=False)
    env = birdseye.env.RFMultiEnv(sensor=sensor, actions=actions, state=state, simulated=False)
    if planner_method in ['dqn', 'DQN']: 
        planner = DQNPlanner(env, device)
    elif planner_method in ['mcts','MCTS']: 
        depth=10
        c=20
        simulations=100
        planner = MCTSPlanner(env, actions, depth, c, simulations)
    belief = env.reset()

    # Flask
    fig = Figure(figsize=(8,8))
    ax = fig.subplots()

    time_step = 0
    run_flask(fig, results)

    data = defaultdict(list)
    time.sleep(2)
    while True: 
        time_step += 1 
        
        action_proposal = actions.index_to_action(planner.proposal(belief))
        action_taken = data.get('action', (0,0))
        
        # update belief based on action and sensor observation (sensor is read inside)
        belief, reward, observation = env.real_step(action_taken, data.get('bearing', None))

        #textstr = ['Actual\nBearing = {:.0f} deg\nSpeed = {:.2f} m/s'.format(data.get('bearing', 0),data.get('distance', 0)), 'Proposed\nBearing = {:.0f} deg\nSpeed = {:.2f} m/s'.format(data.get('bearing',0)+action_proposal[0],action_proposal[1])]        
        textstr = ['Actual\nBearing = {:.0f} deg\nSpeed = {:.2f} m/s'.format(data.get('bearing', 0),action_taken[1]), 'Proposed\nBearing = {:.0f} deg\nSpeed = {:.2f} m/s'.format(data.get('bearing',0)+action_proposal[0],action_proposal[1])]
        results.live_plot(env=env, time_step=time_step, fig=fig, ax=ax, simulated=False, textstr=textstr)
        
        np.save('{}/{}_particles.npy'.format(results.logdir,int(time.time())), env.pf.particles)

        data['action_proposal'].append(action_proposal)
        data['action_taken'].append(action_taken)
        data['reward'].append(reward)
        data['observation'].append(observation)
        
        for key in data: 
            np.save('{}/{}.npy'.format(results.logdir, key), data[key])

if __name__ == '__main__':

    main()

