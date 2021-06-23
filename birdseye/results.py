# results.py

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import re

from .utils import read_header_log
from .definitions import *


def starting_position_plots(config):
    mcts_runs = get_valid_runs('mcts')
    dqn_runs = get_valid_runs('dqn')
    #mcts_config_filter = {'datetime_start': '2021-05-11T02:40:29', 'reward':reward, 'sensor':sensor,  'target_speed':1, 'target_start':['50','150']}
    #dqn_config_filter = {'datetime_start': '2021-05-27T22:55:22', 'reward':reward, 'sensor':sensor, 'target_speed':1, 'target_start':['50','150']}
    mcts_config_filter = {}
    dqn_config_filter = {}
    mcts_config_filter.update(config)
    dqn_config_filter.update(config)
    
    sensor = config.get('sensor', 'all')
    reward = config.get('reward', 'all')
    
    filtered_dqn_runs = sorted(filter_runs('dqn', dqn_runs, dqn_config_filter))
    filtered_mcts_runs = sorted(filter_runs('mcts', mcts_runs, mcts_config_filter))


    fig = plt.figure(figsize=(20,6))
    fig.suptitle('Sensor: {}, Reward: {}'.format(sensor, reward), fontsize=28)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)

    for r in filtered_mcts_runs[-1:]:
        config = get_config('mcts', r)
        data = get_data('mcts', r)
        #print(r,'\n')
        plot_data = list(data['centroid_err'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0]))
        ax1.plot( np.mean(list(plot_data), axis=0), '-', label='mcts spd = {}'.format(float(config['Methods']['target_speed'])))

        ax3.plot( np.mean(list(plot_data), axis=0), '-', label='mcts spd = {}'.format(float(config['Methods']['target_speed'])))



    for r in filtered_dqn_runs[-1:]:
        config = get_config('dqn', r)
        #print(r,'\n')
        #print(config,'\n')
        data = get_data('dqn', r)
        
        plot_data = list(data['centroid_err'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0]))
        y = np.mean(list(plot_data), axis=0)
        ax2.plot( y, ':', label='dqn')
        ax3.plot( y, ':', label='dqn')
        

    ax1.set_xlabel('time step', fontsize=16)
    ax1.set_ylabel('centroid distance', fontsize=16)
    ax1.set_title('MCTS', fontsize=20)
    ax1.legend(fontsize=16)

    ax2.set_xlabel('time step', fontsize=16)
    ax2.set_ylabel('centroid distance', fontsize=16)
    ax2.set_title('DQN', fontsize=20)
    ax2.legend(fontsize=16)

    ax3.set_xlabel('time step', fontsize=16)
    ax3.set_ylabel('centroid distance', fontsize=16)
    ax3.set_title('MCTS vs DQN', fontsize=20)
    ax3.legend(fontsize=16)
    plt.subplots_adjust(
                    hspace=0.4)

    plt.show()

def single_plot(config, metric='centroid_err', variance_bars=False, verbose=False, limit=1):
    reward_str = {'range_reward':'State Dependent Reward', 'entropy_collision_reward':'Belief Dependent Reward'}
    sensor_str = {'drone':'Bearings Sensor','signalstrength':'Signal Strength Sensor'}
    metric_str = {'centroid_err':'Centroid Distance'}
    metric_s = metric_str.get(metric, metric)
    
    mcts_runs = get_valid_runs('mcts')
    dqn_runs = get_valid_runs('dqn')
    mcts_config_filter = {}
    dqn_config_filter = {}
    mcts_config_filter.update(config)
    dqn_config_filter.update(config)

    filtered_dqn_runs = sorted(filter_runs('dqn', dqn_runs, dqn_config_filter))
    filtered_mcts_runs = sorted(filter_runs('mcts', mcts_runs, mcts_config_filter))
    sensor = config.get('sensor', 'all')
    reward = config.get('reward', 'all')

    fig = plt.figure(figsize=(12,8))
    #fig.suptitle('Sensor: {}, Reward: {}'.format(sensor, reward), fontsize=32)

    ax1 = plt.subplot(1,1,1)

    #ax1.set_prop_cycle(color=['red','blue', 'magenta', 'green'])
    ax1.set_prop_cycle(color=['#1f78b4', '#33a02c','#a6cee3','#b2df8a'])
    mcts_avg_inference_time = 10
    dqn_avg_inference_time = 10
    for r in filtered_mcts_runs[-limit:]:
        config = get_config('mcts', r)
        if verbose: 
            print(r,'\n')
            print(config)
            print('=======================')
            
        data = get_data('mcts', r)
        
        
        if data.get('inference_times',None) is not None: 
            mcts_avg_inference_time = np.mean(list(data['inference_times'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))

        plot_data = list(data[metric].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0]))
        y = np.mean(list(plot_data), axis=0)
        
        med = np.percentile(list(plot_data), 50, axis=0)
        low = np.percentile(list(plot_data), 16, axis=0)
        high = np.percentile(list(plot_data), 84, axis=0)
        
        ax1.plot(med, '-', label='MCTS')
        if variance_bars: 
            y_std = np.std(list(plot_data), axis=0)
            ax1.fill_between(np.arange(len(med)), low, high, alpha=0.5)
        print('MCTS, {}, {}, inference time={:.2e}s'.format(sensor_str[config['Methods']['sensor']], reward_str[config['Methods']['reward']], mcts_avg_inference_time))

    for r in filtered_dqn_runs[-limit:]:
        config = get_config('dqn', r)
        
        if verbose: 
            print(r,'\n')
            print(config)
            print('=======================')
            
        data = get_data('dqn', r)
        
        
        if data.get('inference_times',None) is not None: 
            dqn_avg_inference_time = np.mean(list(data['inference_times'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))
            
        plot_data = list(data[metric].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0]))
        # median, 16, 84%
        y = np.mean(list(plot_data), axis=0)
        
        med = np.percentile(list(plot_data), 50, axis=0)
        low = np.percentile(list(plot_data), 16, axis=0)
        high = np.percentile(list(plot_data), 84, axis=0)
        ax1.plot( med, '--', label='DQN')
        
        if variance_bars: 
            y_std = np.std(list(plot_data), axis=0)
            ax1.fill_between(np.arange(len(med)), low, high, alpha=0.5)
    
    # plt caption
        print('DQN, {}, {}, inference time={:.2e}s'.format(sensor_str[config['Methods']['sensor']], reward_str[config['Methods']['reward']], dqn_avg_inference_time))
        

    print('Speedup (MCTS/DQN) = {}x'.format(mcts_avg_inference_time/dqn_avg_inference_time))
    ax1.margins(0)
    
    ax1.set_ylim(0, 125)
    ax1.set_xlabel('Time Step', fontsize=16)
    ax1.set_ylabel('{}'.format(metric_s), fontsize=16)
    #ax1.set_title('{} during single episode'.format(metric), fontsize=24)
    ax1.tick_params(axis='both', which='both', labelsize=14)
    ax1.legend(fontsize=20)


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
                    if config.get(k) is None or float(config.get(k)) != float(v):
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

