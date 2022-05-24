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
from timeit import default_timer as timer

import birdseye.sensor 
import birdseye.env 
import birdseye.actions
import birdseye.state
import birdseye.utils
import birdseye.dqn 
import birdseye.mcts_utils
from birdseye.utils import get_distance, get_bearing

data = {}
data['rssi'] = None
data['position'] = None 
data['distance'] = None 
data['previous_position'] = None 
data['bearing'] = None 
data['previous_bearing'] = None 

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
    def __init__(self, antenna_filename=None, power_tx=26, directivity_tx=1, f=5.7e9, fading_sigma=None, threshold=-120): 
        super().__init__(antenna_filename=antenna_filename, power_tx=26, directivity_tx=1, f=5.7e9, fading_sigma=fading_sigma)
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
        self.runner =  birdseye.mcts_utils.MCTSRunner(env=env, depth=depth, c=c, simulations=simulations)
        self.actions = actions 
    def proposal(self, observation): 
        return self.actions.action_to_index(self.runner.run(observation))

class DQNPlanner(PathPlanner): 
    def __init__(self, env, device, checkpoint_filename): 
        self.model = birdseye.dqn.simple_prep(env, device, checkpoint_filename)
        self.device = device
    def proposal(self, observation): 
        return birdseye.dqn.simple_run(self.model, observation, self.device)
        
# Flask 
def run_flask(fig, results, debug): 
    app = Flask(__name__)

    @app.route("/")
    def hello():
        # Save figure to a temporary buffer.
        flask_start_time = timer()
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        png_filename = '{}/png/{}.png'.format(results.gif_dir, results.time_step)
        #fig.savefig(png_filename, format='png', bbox_inches='tight')
        
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

    host_name = '0.0.0.0'
    port = 4999
    threading.Thread(target=lambda: app.run(host=host_name, port=port, debug=True, use_reloader=False)).start()
    
# main loop 
def main(config=None, debug=False): 

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

    # Sensor 
    antenna_type = 'omni'
    if antenna_type in ['directional', 'yagi', 'logp']: 
        antenna_filename = 'radiation_pattern_yagi_5.csv'
    elif antenna_type in ['omni', 'omnidirectional']: 
        antenna_filename = 'radiation_pattern_monopole.csv'
    power_tx=26
    directivity_tx=1
    f=5.7e9
    sensor = GamutRFSensor(antenna_filename=antenna_filename, power_tx=power_tx, directivity_tx=directivity_tx, f=f, fading_sigma=8, threshold=-120) # fading sigm = 8dB, threshold = -120dB
    
    # Action space 
    actions = WalkingActions()
    actions.print_action_info()
    
    # State managment 
    #reward='entropy_collision_reward'
    reward='heuristic_reward'
    state = birdseye.state.RFMultiState(n_targets=2, reward=reward, simulated=False)
    
    # Environment 
    env = birdseye.env.RFMultiEnv(sensor=sensor, actions=actions, state=state, simulated=False)
    belief = env.reset()

    # Motion planner 
    planner_method = 'mcts'
    if planner_method in ['dqn', 'DQN']: 
        planner = DQNPlanner(env, device, 'checkpoints/dqn_doublerssi.checkpoint')
    elif planner_method in ['mcts','MCTS']: 
        depth=10
        c=20
        simulations=100
        planner = MCTSPlanner(env, actions, depth, c, simulations)
    
    # Flask
    fig = Figure(figsize=(10,10))
    ax = fig.subplots()
    time_step = 0
    run_flask(fig, results, debug)

    # Main loop 
    data = defaultdict(list)
    time.sleep(2)
    while True: 
        loop_start = timer()

        action_start = timer() 
        time_step += 1 
        action_proposal = actions.index_to_action(planner.proposal(belief))
        action_taken = data.get('action', (0,0))
        action_end = timer() 
        
        step_start = timer() 
        # update belief based on action and sensor observation (sensor is read inside)
        belief, reward, observation = env.real_step(action_taken, data.get('bearing', None))
        step_end = timer() 
        

        #textstr = ['Actual\nBearing = {:.0f} deg\nSpeed = {:.2f} m/s'.format(data.get('bearing', 0),data.get('distance', 0)), 'Proposed\nBearing = {:.0f} deg\nSpeed = {:.2f} m/s'.format(data.get('bearing',0)+action_proposal[0],action_proposal[1])]        
        textstr = ['Actual\nBearing = {:.0f} deg\nSpeed = {:.2f} m/s'.format(data.get('bearing', 0),action_taken[1]), 'Proposed\nBearing = {:.0f} deg\nSpeed = {:.2f} m/s'.format(data.get('bearing',0)+action_proposal[0],action_proposal[1])]
        data['position'] = [ 45.598101, -122.678819 ]

        plot_start = timer()
        results.live_plot(env=env, time_step=time_step, fig=fig, ax=ax, data=data, simulated=False, textstr=textstr)
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
    config=None
    main(debug=True)

