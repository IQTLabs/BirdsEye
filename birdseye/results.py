# results.py

import pandas as pd
import numpy as np 
from datetime import datetime

from .utils import read_header_log
from .definitions import *


def get_config(method_name, run_name):
    config = read_header_log('{}/{}/{}_header.txt'.format(RUN_DIR, method_name, run_name))
    return config

def get_data(method_name, run_name):
    data = pd.read_csv('{}/{}/{}_data.csv'.format(RUN_DIR, method_name, run_name))
    append_metric_avgs(data, ['r_err', 'theta_err', 'heading_err', 'centroid_err', 'rmse'])
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
            if k == 'target_speed':
                v = float(v)
                if ((config.get(k) is None) and (v != 1.)) or (float(config.get(k)) != v):
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

