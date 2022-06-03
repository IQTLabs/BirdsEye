# results.py

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import re

from .utils import read_header_log
from .definitions import *


reward_str = {'range_reward':'State Dependent Reward', 'entropy_collision_reward':'Belief Dependent Reward'}
sensor_str = {'drone':'Bearings Sensor','signalstrength':'Signal Strength Sensor'}
metric_str = {'centroid_err':'Centroid Distance'}

def plotter(plot_func, title=None, target_start=78, sensors = ['drone', 'signalstrength'], rewards = ['range_reward', 'entropy_collision_reward'], **kwargs):

    fig = plt.figure(figsize=(20,16), constrained_layout=True)
    fig.suptitle(title)
    subfig_rows = fig.subfigures(nrows=len(sensors), ncols=1)
    axs = [row.subplots(nrows=1, ncols=len(rewards)) for row in subfig_rows]

    for i,s in enumerate(sensors):
        for j,r in enumerate(rewards):
            print('{} & {}'.format(sensor_str[s], reward_str[r]))
            config = {'datetime_start': '2021-06-18T00:00:00', 'reward':r, 'sensor':s, 'target_start': target_start, 'target_speed':1 }
            plot_func(axs[i][j], config, **kwargs)

    plt.show()

def separate_plotter(plot_func, title=None, target_start=78, sensors = ['drone', 'signalstrength'], rewards = ['range_reward', 'entropy_collision_reward'], **kwargs):

    fig = plt.figure(figsize=(20,26), constrained_layout=True)
    fig.suptitle(title)
    subfig_rows = fig.subfigures(nrows=len(sensors)*len(rewards), ncols=1)
    axs = [row.subplots(nrows=1, ncols=2) for row in subfig_rows]

    for i,s in enumerate(sensors):
        for j,r in enumerate(rewards):
            #print('{} & {}'.format(sensor_str[s], reward_str[r]))
            config = {'datetime_start': '2021-06-18T00:00:00', 'reward':r, 'sensor':s, 'target_start': target_start, 'target_speed':1 }
            plot_func(axs[i*len(sensors)+j][0], config, dqn=False, **kwargs)
            plot_func(axs[i*len(sensors)+j][1], config, mcts=False, **kwargs)

    plt.show()

def two_metric_grid(ax1, config, mcts=True, dqn=True, metric1='r_err', metric2='theta_err', variance_bars=False, verbose=False, timing=True, limit=1, y_lim=125):
    # was panel_dual_axis
    # set strings
    reward_str = {'range_reward':'State Dependent Reward', 'entropy_collision_reward':'Belief Dependent Reward'}
    sensor_str = {'drone':'Bearings Sensor','signalstrength':'Signal Strength Sensor'}
    metric_str = {'centroid_err':'Centroid Distance (m)', 'r_err': r'$\delta_{r}$ (m)', 'theta_err':r'$\delta_{\theta}$ (degrees)'}
    #metric_s = metric_str.get(metric, metric)

    # get configs and run data
    mcts_config_filter = {}
    dqn_config_filter = {}
    mcts_config_filter.update(config)
    dqn_config_filter.update(config)
    filtered_dqn_runs = sorted(filter_runs('dqn', dqn_config_filter))
    filtered_mcts_runs = sorted(filter_runs('mcts', mcts_config_filter))
    sensor = config.get('sensor', 'all')
    reward = config.get('reward', 'all')

    ax2=ax1.twinx()

    # blue green color scheme
    if limit==2:
        ax1.set_prop_cycle(color=['#3288bd','#d53e4f'])
        ax2.set_prop_cycle(color=['#66c2a5','#f46d43'])
    else:
        ax1.set_prop_cycle(color=['#1f78b4','#a6cee3', '#33a02c','#b2df8a'])

    mcts_avg_inference_time = 10
    dqn_avg_inference_time = 10
    lns = []
    if mcts:
        for r in filtered_mcts_runs[-limit:]:
            config = get_config('mcts', r)
            if verbose:
                print(r,'\n',config,'\n','========================')

            data = get_data('mcts', r)

            if data.get('inference_times',None) is not None:
                mcts_avg_inference_time = np.mean(list(data['inference_times'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))


            ############################

            plot_data1 = np.abs(list(data[metric1].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))
            med1 = np.percentile(list(plot_data1), 50, axis=0)
            low1 = np.percentile(list(plot_data1), 16, axis=0)
            high1 = np.percentile(list(plot_data1), 84, axis=0)

            l1 = ax1.plot(med1, '-', label='MCTS, '+metric_str[metric1])

            plot_data2 = np.abs(list(data[metric2].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))
            med2 = np.percentile(list(plot_data2), 50, axis=0)
            low2 = np.percentile(list(plot_data2), 16, axis=0)
            high2 = np.percentile(list(plot_data2), 84, axis=0)

            l2 = ax2.plot(med2, '-', label='MCTS, '+metric_str[metric2])
            if variance_bars:
                ax1.fill_between(np.arange(len(med1)), low1, high1, alpha=0.2)
                ax2.fill_between(np.arange(len(med2)), low2, high2, alpha=0.2)

            lns += l1+l2
            ############################

            ###########################3
            #print('MCTS, {}, {}, inference time={:.2e}s'.format(sensor_str[config['Methods']['sensor']], reward_str[config['Methods']['reward']], mcts_avg_inference_time))
            if timing:
                print('MCTS inference time={:.2e}s'.format(mcts_avg_inference_time))
    if dqn:
        for r in filtered_dqn_runs[-limit:]:
            config = get_config('dqn', r)
            if verbose:
                print(r,'\n',config,'\n','========================')

            data = get_data('dqn', r)


            if data.get('inference_times',None) is not None:
                dqn_avg_inference_time = np.mean(list(data['inference_times'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))

            plot_data1 = np.abs(list(data[metric1].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))
            med1 = np.percentile(list(plot_data1), 50, axis=0)
            low1 = np.percentile(list(plot_data1), 16, axis=0)
            high1 = np.percentile(list(plot_data1), 84, axis=0)

            l3 = ax1.plot(med1, '--', label='DQN, '+metric_str[metric1])

            plot_data2 = np.abs(list(data[metric2].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))
            med2 = np.percentile(list(plot_data2), 50, axis=0)
            low2 = np.percentile(list(plot_data2), 16, axis=0)
            high2 = np.percentile(list(plot_data2), 84, axis=0)

            l4 = ax2.plot(med2, '--', label='DQN, '+metric_str[metric2])
            if variance_bars:
                ax1.fill_between(np.arange(len(med1)), low1, high1, alpha=0.2)
                ax2.fill_between(np.arange(len(med2)), low2, high2, alpha=0.2)

            lns += l3+l4

            #print('DQN, {}, {}, inference time={:.2e}s'.format(sensor_str[config['Methods']['sensor']], reward_str[config['Methods']['reward']], dqn_avg_inference_time))
            if timing:
                print('DQN inference time={:.2e}s'.format(dqn_avg_inference_time))

    if mcts and dqn and timing:
        print('Speedup (MCTS/DQN) = {:.2f}x'.format(mcts_avg_inference_time/dqn_avg_inference_time))
        print('======================================')
    ax1.margins(0)

    #ax1.set_ylim(0, y_lim)
    ax1.set_xlabel('Time Step', fontsize=16)
    ax1.set_ylabel(metric_str[metric1], fontsize=24)
    ax2.set_ylabel(metric_str[metric2], fontsize=24)
    ax1.tick_params(axis='both', which='both', labelsize=14)
    ax2.tick_params(axis='both', which='both', labelsize=14)

    ax1.set_title('{} & {}'.format(sensor_str[sensor], reward_str[reward]))

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=20)

def single_std_dev(ax1, config, metric='r', variance_bars=False, verbose=False, limit=1, y_lim=125):
    # was single_plot_var
    cov_idx = {'r':0, 'theta':1}

    reward_str = {'range_reward':'State Dependent Reward', 'entropy_collision_reward':'Belief Dependent Reward'}
    reward_labels = {'range_reward':'state', 'entropy_collision_reward':'belief'}
    sensor_str = {'drone':'Bearings Sensor','signalstrength':'Signal Strength Sensor'}
    metric_str = {'r':'r','theta':'\\theta'}
    metric_s = metric_str.get(metric, metric)

    label_str = ''

    mcts_config_filter = {}
    dqn_config_filter = {}
    mcts_config_filter.update(config)
    dqn_config_filter.update(config)

    filtered_dqn_runs = sorted(filter_runs('dqn', dqn_config_filter))
    filtered_mcts_runs = sorted(filter_runs('mcts', mcts_config_filter))
    sensor = config.get('sensor', 'all')
    reward = config.get('reward', 'all')

    # red blue color scheme
    if limit==1:
        ax1.set_prop_cycle(color=['#ca0020', '#0571b0'])
    else:
        ax1.set_prop_cycle(color=['#ca0020','#f4a582', '#0571b0','#92c5de'])

    # blue green color scheme
    if limit==1:
        ax1.set_prop_cycle(color=['#1f78b4', '#33a02c'])
    elif limit==2:
        ax1.set_prop_cycle(color=['#3288bd','#66c2a5','#d53e4f','#f46d43'])
    else:
        ax1.set_prop_cycle(color=['#1f78b4','#a6cee3', '#33a02c','#b2df8a'])

    mcts_avg_inference_time = 10
    dqn_avg_inference_time = 10
    lns = []
    for r in filtered_mcts_runs[-limit:]:
        config = get_config('mcts', r)
        if verbose:
            print(r,'\n')
            print(config)
            print('=======================')

        data = get_data('mcts', r)

        if data.get('inference_times',None) is not None:
            mcts_avg_inference_time = np.mean(list(data['inference_times'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))

        pf_cov = data.get('pf_cov',None)
        if pf_cov is not None:
            pf_cov = np.array([eval(cov_str) for cov_str in pf_cov])
            pf_cov = pf_cov.reshape(pf_cov.shape[0], pf_cov.shape[1], 4, 4)

        else:
            break

        variance = np.sqrt(pf_cov[:,:,cov_idx[metric],cov_idx[metric]])

        variance_med = np.percentile(variance, 50, axis=0)
        variance_low = np.percentile(variance, 16, axis=0)
        variance_high = np.percentile(variance, 84, axis=0)


        if limit == 2:
            label_str = r', $R_{{\mathrm{{{}}}}}$'.format(reward_labels[config['Methods']['reward']])
        ax1.plot(variance_med, '-', label='MCTS'+'{}'.format(label_str))
        if variance_bars:
            ax1.fill_between(np.arange(len(variance_med)), variance_low, variance_high, alpha=0.2)
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

        pf_cov = data.get('pf_cov',None)
        if pf_cov is not None:
            pf_cov = np.array([eval(cov_str) for cov_str in pf_cov])
            pf_cov = pf_cov.reshape(pf_cov.shape[0], pf_cov.shape[1], 4, 4)

        else:
            break

        variance = np.sqrt(pf_cov[:,:,cov_idx[metric],cov_idx[metric]])

        variance_med = np.percentile(variance, 50, axis=0)
        variance_low = np.percentile(variance, 16, axis=0)
        variance_high = np.percentile(variance, 84, axis=0)

        if limit == 2:
            label_str = r', $R_{{\mathrm{{{}}}}}$'.format(reward_labels[config['Methods']['reward']])
        ax1.plot(variance_med, '--', label='DQN'+'{}'.format(label_str))
        if variance_bars:
            ax1.fill_between(np.arange(len(variance_med)), variance_low, variance_high, alpha=0.2)

    # plt caption
        print('DQN, {}, {}, inference time={:.2e}s'.format(sensor_str[config['Methods']['sensor']], reward_str[config['Methods']['reward']], dqn_avg_inference_time))


    print('Speedup (MCTS/DQN) = {:.2f}x'.format(mcts_avg_inference_time/dqn_avg_inference_time))
    print('======================================')
    ax1.margins(0)

    #ax1.set_ylim(0, y_lim)
    ax1.set_xlabel('Time Step', fontsize=16)
    ax1.set_ylabel(r'$\sigma_{{{}}}$'.format(metric_s), fontsize=22)
    #ax1.set_title('{} during single episode'.format(metric), fontsize=24)
    ax1.tick_params(axis='both', which='both', labelsize=14)
    ax1.legend(fontsize=20)


def std_dev_grid(ax1, config, mcts=True, dqn=True, variance_bars=False, verbose=False, timing=True, limit=1, y_lim=125):
    # was single_plot_combined_cov
    reward_str = {'range_reward':'State Dependent Reward', 'entropy_collision_reward':'Belief Dependent Reward'}
    sensor_str = {'drone':'Bearings Sensor','signalstrength':'Signal Strength Sensor'}
    #metric_str = {'centroid_err':'Centroid Distance'}
    #metric_s = metric_str.get(metric, metric)

    mcts_config_filter = {}
    dqn_config_filter = {}
    mcts_config_filter.update(config)
    dqn_config_filter.update(config)

    filtered_dqn_runs = sorted(filter_runs('dqn', dqn_config_filter))
    filtered_mcts_runs = sorted(filter_runs('mcts', mcts_config_filter))
    sensor = config.get('sensor', 'all')
    reward = config.get('reward', 'all')


    ax2=ax1.twinx()

    #ax1.set_prop_cycle(color=['red','blue', 'magenta', 'green'])

    # blue green color scheme
    if limit==1:
        ax1.set_prop_cycle(color=['#1f78b4', '#33a02c'])
    elif limit==2:
        ax1.set_prop_cycle(color=['#3288bd','#d53e4f'])
        ax2.set_prop_cycle(color=['#66c2a5','#f46d43'])
    else:
        ax1.set_prop_cycle(color=['#1f78b4','#a6cee3', '#33a02c','#b2df8a'])

    mcts_avg_inference_time = 10
    dqn_avg_inference_time = 10
    lns = []

    if mcts:
        for r in filtered_mcts_runs[-limit:]:
            config = get_config('mcts', r)
            if verbose:
                print(r,'\n')
                print(config)
                print('=======================')

            data = get_data('mcts', r)

            if data.get('inference_times',None) is not None:
                mcts_avg_inference_time = np.mean(list(data['inference_times'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))

            pf_cov = data.get('pf_cov',None)
            if pf_cov is not None:
                pf_cov = np.array([eval(cov_str) for cov_str in pf_cov])
                pf_cov = pf_cov.reshape(pf_cov.shape[0], pf_cov.shape[1], 4, 4)

            else:
                break

            r_var = np.sqrt(pf_cov[:,:,0,0])
            theta_var = np.sqrt(pf_cov[:,:,1,1])

            r_med = np.percentile(r_var, 50, axis=0)
            r_low = np.percentile(r_var, 16, axis=0)
            r_high = np.percentile(r_var, 84, axis=0)

            theta_med = np.percentile(theta_var, 50, axis=0)
            theta_low = np.percentile(theta_var, 16, axis=0)
            theta_high = np.percentile(theta_var, 84, axis=0)

            l1= ax1.plot(r_med, '-', label=r'MCTS, $\sigma_{r}$')
            l2= ax2.plot(theta_med, '-', label=r'MCTS, $\sigma_{\theta}$')
            lns += l1+l2
            if variance_bars:
                ax1.fill_between(np.arange(len(r_med)), r_low, r_high, alpha=0.2)
                ax2.fill_between(np.arange(len(theta_med)), theta_low, theta_high, alpha=0.2)
            #print('MCTS, {}, {}, inference time={:.2e}s'.format(sensor_str[config['Methods']['sensor']], reward_str[config['Methods']['reward']], mcts_avg_inference_time))
            if timing:
                print('MCTS inference time={:.2e}s'.format(mcts_avg_inference_time))
    if dqn:
        for r in filtered_dqn_runs[-limit:]:
            config = get_config('dqn', r)

            if verbose:
                print(r,'\n')
                print(config)
                print('=======================')

            data = get_data('dqn', r)


            if data.get('inference_times',None) is not None:
                dqn_avg_inference_time = np.mean(list(data['inference_times'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))

            pf_cov = data.get('pf_cov',None)
            if pf_cov is not None:
                pf_cov = np.array([eval(cov_str) for cov_str in pf_cov])
                pf_cov = pf_cov.reshape(pf_cov.shape[0], pf_cov.shape[1], 4, 4)

            else:
                break

            r_var = np.sqrt(pf_cov[:,:,0,0])
            theta_var = np.sqrt(pf_cov[:,:,1,1])

            r_med = np.percentile(r_var, 50, axis=0)
            r_low = np.percentile(r_var, 16, axis=0)
            r_high = np.percentile(r_var, 84, axis=0)

            theta_med = np.percentile(theta_var, 50, axis=0)
            theta_low = np.percentile(theta_var, 16, axis=0)
            theta_high = np.percentile(theta_var, 84, axis=0)

            l3=ax1.plot(r_med, '--', label=r'DQN, $\sigma_{r}$')
            l4=ax2.plot(theta_med, '--', label=r'DQN, $\sigma_{\theta}$')
            lns += l3+l4

            if variance_bars:
                ax1.fill_between(np.arange(len(r_med)), r_low, r_high, alpha=0.2)
                ax2.fill_between(np.arange(len(theta_med)), theta_low, theta_high, alpha=0.2)
            #print('DQN, {}, {}, inference time={:.2e}s'.format(sensor_str[config['Methods']['sensor']], reward_str[config['Methods']['reward']], dqn_avg_inference_time))
            if timing:
                print('DQN inference time={:.2e}s'.format(dqn_avg_inference_time))

    if mcts and dqn and timing:
        print('Speedup (MCTS/DQN) = {:.2f}x'.format(mcts_avg_inference_time/dqn_avg_inference_time))
        print('======================================')
    ax1.margins(0)

    #ax1.set_ylim(0, y_lim)
    ax1.set_xlabel('Time Step', fontsize=16)
    ax1.set_ylabel(r'$\sigma_{r}$ (m)', fontsize=24)
    ax2.set_ylabel(r'$\sigma_{\theta}$ (degrees)', fontsize=24)
    ax1.tick_params(axis='both', which='both', labelsize=14)
    ax2.tick_params(axis='both', which='both', labelsize=14)
    ax1.set_title('{} & {}'.format(sensor_str[sensor], reward_str[reward]))
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=20)



def single_metric_grid(ax1, config, metric='centroid_err', variance_bars=False, verbose=False, limit=1, y_lim=125):
    # was single_plot_combined
    reward_str = {'range_reward':'State Dependent Reward', 'entropy_collision_reward':'Belief Dependent Reward'}
    sensor_str = {'drone':'Bearings Sensor','signalstrength':'Signal Strength Sensor'}
    metric_str = {'centroid_err':'Centroid Distance (m)'}
    metric_s = metric_str.get(metric, metric)

    mcts_config_filter = {}
    dqn_config_filter = {}
    mcts_config_filter.update(config)
    dqn_config_filter.update(config)

    filtered_dqn_runs = sorted(filter_runs('dqn', dqn_config_filter))
    filtered_mcts_runs = sorted(filter_runs('mcts', mcts_config_filter))

    sensor = config.get('sensor', 'all')
    reward = config.get('reward', 'all')

    #ax1.set_prop_cycle(color=['red','blue', 'magenta', 'green'])

    # red blue color scheme
    if limit==1:
        ax1.set_prop_cycle(color=['#ca0020', '#0571b0'])
    else:
        ax1.set_prop_cycle(color=['#ca0020','#f4a582', '#0571b0','#92c5de'])

    # blue green color scheme
    if limit==1:
        ax1.set_prop_cycle(color=['#1f78b4', '#33a02c'])
    else:
        ax1.set_prop_cycle(color=['#1f78b4','#a6cee3', '#33a02c','#b2df8a'])
    mcts_avg_inference_time = 10
    dqn_avg_inference_time = 10
    for r in filtered_mcts_runs[-limit:]:
        config = get_config('mcts', r)

        data = get_data('mcts', r)
        if verbose:
            print(r,'\n')
            print(config)
            print('=======================')


        if data.get('inference_times',None) is not None:
            mcts_avg_inference_time = np.mean(list(data['inference_times'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))

        plot_data = np.abs(list(data[metric].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))
        y = np.mean(list(plot_data), axis=0)
        if verbose:
            print(r,'\n')
            print(config)
            print(y)
            print('=======================')

        med = np.percentile(list(plot_data), 50, axis=0)
        low = np.percentile(list(plot_data), 16, axis=0)
        high = np.percentile(list(plot_data), 84, axis=0)

        ax1.plot(med, '-', label='MCTS')
        if variance_bars:
            y_std = np.std(list(plot_data), axis=0)
            ax1.fill_between(np.arange(len(med)), low, high, alpha=0.2)
        #print('run: {}'.format(r))
        #print('MCTS, {}, {}, inference time={:.2e}s'.format(sensor_str[config['Methods']['sensor']], reward_str[config['Methods']['reward']], mcts_avg_inference_time))
        print('MCTS inference time={:.2e}s'.format(mcts_avg_inference_time))

    for r in filtered_dqn_runs[-limit:]:
        config = get_config('dqn', r)

        if verbose:
            print(r,'\n')
            print(config)
            print('=======================')

        data = get_data('dqn', r)


        if data.get('inference_times',None) is not None:
            dqn_avg_inference_time = np.mean(list(data['inference_times'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))

        plot_data = np.abs(list(data[metric].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0])))
        # median, 16, 84%
        y = np.mean(list(plot_data), axis=0)

        med = np.percentile(list(plot_data), 50, axis=0)
        low = np.percentile(list(plot_data), 16, axis=0)
        high = np.percentile(list(plot_data), 84, axis=0)
        ax1.plot( med, '--', label='DQN')

        if variance_bars:
            y_std = np.std(list(plot_data), axis=0)
            ax1.fill_between(np.arange(len(med)), low, high, alpha=0.2)

    # plt caption
        #print('run: {}'.format(r))
        #print('DQN, {}, {}, inference time={:.2e}s'.format(sensor_str[config['Methods']['sensor']], reward_str[config['Methods']['reward']], dqn_avg_inference_time))
        print('DQN inference time={:.2e}s'.format(dqn_avg_inference_time))


    print('Speedup (MCTS/DQN) = {:.2f}x'.format(mcts_avg_inference_time/dqn_avg_inference_time))
    print('======================================')
    ax1.margins(0)

    ax1.set_ylim(0, y_lim)
    ax1.set_xlabel('Time Step', fontsize=16)
    ax1.set_ylabel('{}'.format(metric_s), fontsize=16)
    #ax1.set_title('{} during single episode'.format(metric), fontsize=24)
    ax1.tick_params(axis='both', which='both', labelsize=14)
    ax1.legend(fontsize=20)
    ax1.set_title('{} & {}'.format(sensor_str[sensor], reward_str[reward]))


def starting_position_plots(config, limit=1, metric='centroid_err'):

    reward_str = {'range_reward':'State Dependent Reward', 'entropy_collision_reward':'Belief Dependent Reward'}
    sensor_str = {'drone':'Bearings Sensor','signalstrength':'Signal Strength Sensor'}
    metric_str = {'centroid_err':'Centroid Distance (m)'}
    metric_s = metric_str.get(metric, metric)

    #mcts_config_filter = {'datetime_start': '2021-05-11T02:40:29', 'reward':reward, 'sensor':sensor,  'target_speed':1, 'target_start':['50','150']}
    #dqn_config_filter = {'datetime_start': '2021-05-27T22:55:22', 'reward':reward, 'sensor':sensor, 'target_speed':1, 'target_start':['50','150']}
    mcts_config_filter = {}
    dqn_config_filter = {}
    mcts_config_filter.update(config)
    dqn_config_filter.update(config)

    sensor = config.get('sensor', 'all')
    reward = config.get('reward', 'all')

    filtered_dqn_runs = sorted(filter_runs('dqn', dqn_config_filter))
    filtered_mcts_runs = sorted(filter_runs('mcts', mcts_config_filter))

    sorted_filtered_mcts_runs = sorted(filtered_mcts_runs[-limit:], key=lambda r: int(get_config('mcts', r)['Methods']['target_start']), reverse=True)
    sorted_filtered_dqn_runs = sorted(filtered_dqn_runs[-limit:], key=lambda r: int(get_config('dqn', r)['Methods']['target_start']), reverse=True)

    fig = plt.figure(figsize=(20,6))
    #fig.suptitle('{} & {}'.format(sensor_str[sensor], reward_str[reward]), fontsize=28)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)

    if limit==3:
        ax1.set_prop_cycle(color=['#3288bd','#66c2a5','#abdda4'])
        ax2.set_prop_cycle(color=['#d53e4f','#f46d43','#fdae61'])
        ax3.set_prop_cycle(color=['#3288bd','#66c2a5','#abdda4','#d53e4f','#f46d43','#fdae61'])

    for r in sorted_filtered_mcts_runs:
        config = get_config('mcts', r)
        data = get_data('mcts', r)
        #print(r,'\n')
        plot_data = list(data['centroid_err'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0]))
        target_start = int(config['Methods']['target_start'])
        ax1.plot( np.mean(list(plot_data), axis=0), '-', label=r'$r_0 \in [{},{}]$'.format(target_start-25, target_start+25))
        ax3.plot( np.mean(list(plot_data), axis=0), '-', label=r'MCTS, $r_0 \in [{},{}]$'.format(target_start-25, target_start+25))

        #print('MCTS, {}, {}, target_start = {}'.format(sensor_str[config['Methods']['sensor']], reward_str[config['Methods']['reward']], config['Methods']['target_start']))


    for r in sorted_filtered_dqn_runs:
        config = get_config('dqn', r)
        #print(r,'\n')
        #print(config,'\n')
        data = get_data('dqn', r)

        plot_data = list(data['centroid_err'].apply(lambda x: [float(xx) for xx in re.split(', |\s+', x[1:-1]) if len(xx) > 0]))
        y = np.mean(list(plot_data), axis=0)
        target_start = int(config['Methods']['target_start'])
        ax2.plot( y, '-', label=r'$r_0 \in [{},{}]$'.format(target_start-25, target_start+25))
        ax3.plot( y, '--', label=r'DQN, $r_0 \in [{},{}]$'.format(target_start-25, target_start+25))

        #print('DQN, {}, {}, target_start = {}'.format(sensor_str[config['Methods']['sensor']], reward_str[config['Methods']['reward']], config['Methods']['target_start']))

    ax1.margins(0)
    ax1.set_ylim(0, 125)
    ax1.set_xlabel('Time Step', fontsize=16)
    ax1.set_ylabel('{}'.format(metric_s), fontsize=16)
    ax1.set_title('MCTS', fontsize=20)
    ax1.legend(fontsize=10)

    ax2.margins(0)
    ax2.set_ylim(0, 125)
    ax2.set_xlabel('Time Step', fontsize=16)
    ax2.set_ylabel('{}'.format(metric_s), fontsize=16)
    ax2.set_title('DQN', fontsize=20)
    ax2.legend(fontsize=10)

    ax3.margins(0)
    ax3.set_ylim(0, 125)
    ax3.set_xlabel('Time Step', fontsize=16)
    ax3.set_ylabel('{}'.format(metric_s), fontsize=16)
    ax3.set_title('MCTS vs DQN', fontsize=20)
    ax3.legend(fontsize=10)
    #plt.subplots_adjust(
    #                hspace=0.4)
    plt.subplots_adjust(top=0.85)
    plt.suptitle('{} & {}'.format(sensor_str[sensor], reward_str[reward]))

    plt.show()

def single_plot(config, metric='centroid_err', variance_bars=False, verbose=False, limit=1):
    reward_str = {'range_reward':'State Dependent Reward', 'entropy_collision_reward':'Belief Dependent Reward'}
    reward_labels = {'range_reward':'state', 'entropy_collision_reward':'belief'}
    sensor_str = {'drone':'Bearings Sensor','signalstrength':'Signal Strength Sensor'}
    metric_str = {'centroid_err':'Centroid Distance'}
    metric_s = metric_str.get(metric, metric)

    mcts_config_filter = {}
    dqn_config_filter = {}
    mcts_config_filter.update(config)
    dqn_config_filter.update(config)

    filtered_dqn_runs = sorted(filter_runs('dqn', dqn_config_filter))
    filtered_mcts_runs = sorted(filter_runs('mcts', mcts_config_filter))
    sensor = config.get('sensor', 'all')
    reward = config.get('reward', 'all')

    fig = plt.figure(figsize=(12,8))
    #fig.suptitle('Sensor: {}, Reward: {}'.format(sensor, reward), fontsize=32)

    ax1 = plt.subplot(1,1,1)


    #ax1.set_prop_cycle(color=['red','blue', 'magenta', 'green'])

    # red blue color scheme
    if limit==1:
        ax1.set_prop_cycle(color=['#ca0020', '#0571b0'])
    else:
        ax1.set_prop_cycle(color=['#ca0020','#f4a582', '#0571b0','#92c5de'])

    # blue green color scheme
    if limit==1:
        ax1.set_prop_cycle(color=['#1f78b4', '#33a02c'])
    elif limit==2:
        ax1.set_prop_cycle(color=['#3288bd','#66c2a5','#d53e4f','#f46d43'])
    else:
        ax1.set_prop_cycle(color=['#1f78b4','#a6cee3', '#33a02c','#b2df8a'])
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

        mcts_label_str = 'MCTS'
        if limit==2:
            mcts_label_str = r'MCTS, $R_{{\mathrm{{{}}}}}$'.format(reward_labels[config['Methods']['reward']])
        ax1.plot(med, '-', label=mcts_label_str)
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

        dqn_label_str = 'DQN'
        if limit==2:
            dqn_label_str = r'DQN, $R_{{\mathrm{{{}}}}}$'.format(reward_labels[config['Methods']['reward']])

        ax1.plot( med, '--', label=dqn_label_str)

        if variance_bars:
            y_std = np.std(list(plot_data), axis=0)
            ax1.fill_between(np.arange(len(med)), low, high, alpha=0.5)

    # plt caption
        print('DQN, {}, {}, inference time={:.2e}s'.format(sensor_str[config['Methods']['sensor']], reward_str[config['Methods']['reward']], dqn_avg_inference_time))


    print('Speedup (MCTS/DQN) = {:.2f}x'.format(mcts_avg_inference_time/dqn_avg_inference_time))
    ax1.margins(0)

    ax1.set_ylim(0, 125)
    ax1.set_xlabel('Time Step', fontsize=16)
    ax1.set_ylabel('{}'.format(metric_s), fontsize=16)
    #ax1.set_title('{} during single episode'.format(metric), fontsize=24)
    ax1.tick_params(axis='both', which='both', labelsize=14)
    ax1.legend(fontsize=20)

    plt.show()


### Results file reader functions
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

def filter_runs(method_name, config_filter=None):

    runs = get_valid_runs(method_name)

    filtered_runs = []

    for r in runs:
        match = True
        if method_name == 'baseline':
            config = get_config(method_name, r)['Methods']
            config.update(get_config(method_name, r)['Defaults'])
        else:
            config = get_config(method_name, r)['Methods']
        for k,v in config_filter.items():
            #if v is None:
            #    continue
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
            elif k == 'fading_sigma':
                v = float(v)
                if float(config.get(k,0.0)) != v:
                    match = False
                    break
            elif k == 'particle_resample':
                v = float(v)
                if float(config.get(k,0.005)) != v:
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
    filtered_dqn_runs = filter_runs('dqn', config_filter)
    print(filtered_dqn_runs)
    print(dqn_runs)
    #print(mcts_runs)
    run_name = '2021-04-21T09:46:52'

    config = get_config(method_name, run_name)
    data = get_data(method_name, run_name)

    #print(config)
    #print(data)

