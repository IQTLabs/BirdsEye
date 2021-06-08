# results.py

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import re

from .utils import read_header_log
from .definitions import *


def starting_position_plots(reward, sensor): 
    mcts_runs = get_valid_runs('mcts')
    dqn_runs = get_valid_runs('dqn')
    mcts_config_filter = {'datetime_start': '2021-05-11T02:40:29', 'reward':reward, 'sensor':sensor,  'target_speed':1, 'target_start':['50','150']}
    dqn_config_filter = {'datetime_start': '2021-05-27T22:55:22', 'reward':reward, 'sensor':sensor, 'target_speed':1, 'target_start':['50','150']}
    filtered_dqn_runs = filter_runs('dqn', dqn_runs, dqn_config_filter)
    filtered_mcts_runs = filter_runs('mcts', mcts_runs, mcts_config_filter)


    fig = plt.figure(figsize=(20,6))
    fig.suptitle('Sensor: {}, Reward: {}'.format(sensor, reward))

    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)

    for r in filtered_mcts_runs: 
        config = get_config('mcts', r)
        data = get_data('mcts', r)
        #print(r,'\n')
        plot_data = list(data['centroid_err'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0]))
        ax1.plot( np.mean(list(plot_data), axis=0), '-', label='mcts spd = {}'.format(float(config['Methods']['target_speed'])))
        
        ax3.plot( np.mean(list(plot_data), axis=0), '-', label='mcts spd = {}'.format(float(config['Methods']['target_speed'])))
        
    
    
    for r in filtered_dqn_runs: 
        config = get_config('dqn', r)
        #print(r,'\n')
        #print(config,'\n')
        data = get_data('dqn', r)
        plot_data = data['centroid_err'].iloc[-1]
        if type(plot_data) != np.float64: 
            plot_data = [float(xx) for xx in re.split(', |\s+', plot_data.strip('[]\n'))]
            ax2.plot( plot_data, ':', label='dqn spd = {}'.format(float(config['Methods']['target_speed'])))
            ax3.plot( plot_data, ':', label='dqn spd = {}'.format(float(config['Methods']['target_speed'])))
        
        
    ax1.set_xlabel('time step')
    ax1.set_ylabel('centroid distance')
    ax1.set_title('MCTS')
    ax1.legend()
    
    ax2.set_xlabel('time step')
    ax2.set_ylabel('centroid distance')
    ax2.set_title('DQN')
    ax2.legend()
    
    ax3.set_xlabel('time step')
    ax3.set_ylabel('centroid distance')
    ax3.set_title('MCTS vs DQN')
    ax3.legend()


    plt.show()
    
def single_plot(config): 
    mcts_runs = get_valid_runs('mcts')
    dqn_runs = get_valid_runs('dqn')
    mcts_config_filter = {'datetime_start': '2021-05-11T02:40:29'}
    dqn_config_filter = {'datetime_start': '2021-05-27T22:55:22'}
    mcts_config_filter.update(config)
    dqn_config_filter.update(config)
    
    filtered_dqn_runs = filter_runs('dqn', dqn_runs, dqn_config_filter)
    filtered_mcts_runs = filter_runs('mcts', mcts_runs, mcts_config_filter)
    sensor = config.get('sensor', 'all')
    reward = config.get('reward', 'all')

    fig = plt.figure(figsize=(20,8))
    fig.suptitle('Sensor: {}, Reward: {}'.format(sensor, reward))

    ax1 = plt.subplot(1,1,1)

    for r in filtered_mcts_runs: 
        config = get_config('mcts', r)
        data = get_data('mcts', r)
        #print(r,'\n')
        plot_data = list(data['centroid_err'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0]))
        ax1.plot( np.mean(list(plot_data), axis=0), '-', label='mcts, {}, {}'.format(config['Methods']['sensor'], config['Methods']['reward']))
       
    

    for r in filtered_dqn_runs: 
        config = get_config('dqn', r)
        #print(r,'\n')
        #print(config,'\n')
        data = get_data('dqn', r)
        plot_data = data['centroid_err'].iloc[-1]
        if type(plot_data) != np.float64: 
            plot_data = [float(xx) for xx in re.split(', |\s+', plot_data.strip('[]\n'))]
            ax1.plot( plot_data, ':', label='dqn, {}, {}'.format(config['Methods']['sensor'], config['Methods']['reward']))
        
        
    
    ax1.set_xlabel('time step')
    ax1.set_ylabel('centroid distance')
    ax1.set_title('centroid distance during single episode')
    ax1.legend()


    plt.show()

def get_config(method_name, run_name):
    config = read_header_log('{}/{}/{}_header.txt'.format(RUN_DIR, method_name, run_name))
    config['Methods']['reward'] = config['Methods'].get('reward', 'range_reward')
    return config

def get_data(method_name, run_name):
    data = pd.read_csv('{}/{}/{}_data.csv'.format(RUN_DIR, method_name, run_name))
    #append_metric_avgs(data, ['r_err', 'theta_err', 'heading_err', 'centroid_err', 'rmse'])
    return data

def append_metric_avgs(df, metrics):

    for m in metrics:
        if ('avg_{}'.format(m) not in df) and ('average_{}'.format(m) not in df):
            df['avg_{}'.format(m)] = np.mean(list(df[m]), axis=1)

def get_valid_runs(method_name):
    files = os.listdir('{}/{}/'.format(RUN_DIR, method_name))
    runs = list(set([f.split('_')[0] for f in files]))
    valid_runs = []
    for r in runs:
        if (r+'_data.csv' in files) and (r+'_header.txt' in files):
            try:
                get_config(method_name, r)
                valid_runs.append(r)
            except:
                pass
    return valid_runs

def filter_runs(method_name, runs, config_filter=None):

    filtered_runs = []

    for r in runs:
        match = True
        config = get_config(method_name, r)['Methods']
        for k,v in config_filter.items():
            if v is None:
                continue
            if k == 'target_speed':
                v = float(v)
                if ((config.get(k) is None) and (v != 1.)) or (float(config.get(k)) != v):
                    match = False
                    break
            elif k == 'target_start': 
                if type(v) == list: 
                    if config.get(k, None) not in v: 
                        match = False
                        break
                else: 
                    if config.get(k) != v: 
                        match = False
                        break
            elif k == 'datetime_start':
                config_datetime = datetime.strptime(v, '%Y-%m-%dT%H:%M:%S')
                run_datetime = datetime.strptime(r, '%Y-%m-%dT%H:%M:%S')
                if run_datetime < config_datetime:
                    match = False
                    break
            elif k == 'datetime_end':
                config_datetime = datetime.strptime(v, '%Y-%m-%dT%H:%M:%S')
                run_datetime = datetime.strptime(r, '%Y-%m-%dT%H:%M:%S')
                if run_datetime > config_datetime:
                    match = False
                    break
            elif config.get(k) != v:
                match = False
                break
        if match:
            filtered_runs.append(r)

    return filtered_runs



def show_results():
    method_name = 'mcts' # should be 'mcts' or 'dqn'

    dqn_runs = get_valid_runs('dqn')
    mcts_runs = get_valid_runs('mcts')


    config_filter = {'method':'dqn'}
    filtered_dqn_runs = filter_runs('dqn', dqn_runs, config_filter)
    print(filtered_dqn_runs)
    print(dqn_runs)
    #print(mcts_runs)
    run_name = '2021-04-21T09:46:52'

    config = get_config(method_name, run_name)
    data = get_data(method_name, run_name)

    #print(config)
    #print(data)

