# utils.py

import numpy as np
import json
import pandas as pd
from pathlib import Path
import imageio

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from IPython.display import clear_output
from scipy.ndimage.filters import gaussian_filter

from .definitions import *

##################################################################
# Transforms
##################################################################
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

##################################################################
# Saving Results
##################################################################
class Results(object):
    '''
    Results class for saving run results
    to file with common format.
    '''
    def __init__(self, method_name='', global_start_time='', num_iters=0, plotting=False):
        self.num_iters = num_iters
        self.method_name = method_name
        self.global_start_time = global_start_time
        self.plotting = plotting
        if type(self.plotting) != bool:
            if self.plotting == 'true' or self.plotting == 'True':
                self.plotting = True
            else:
                self.plotting = False
        self.namefile = '{}/{}/{}_data.csv'.format(RUN_DIR, method_name, global_start_time)
        self.gif_dir = '{}/{}/{}'.format(RUN_DIR, method_name, global_start_time)

        Path(self.gif_dir+'/png/').mkdir(parents=True, exist_ok=True)
        Path(self.gif_dir+'/gif/').mkdir(parents=True, exist_ok=True)
        self.col_names =['time', 'run_time', 'target_state', 'sensor_state',
                         'action', 'observation', 'reward', 'collisions', 'lost',
                         'r_err', 'theta_err', 'heading_err', 'centroid_err', 'rmse','mae','inference_times', 'pf_cov']

        self.abs_target_hist = []
        self.abs_sensor_hist = []
        self.history_length = 50

    # Save dataframe to CSV file
    def write_dataframe(self, run_data):
        if os.path.isfile(self.namefile):
            print('Updating file {}'.format(self.namefile))
        else:
            print('Saving file to {}'.format(self.namefile))
        df = pd.DataFrame(run_data, columns=self.col_names)
        df.to_csv(self.namefile)


    def save_gif(self, run, sub_run=None):
        filename = run if sub_run is None else '{}_{}'.format(run, sub_run)
        # Build GIF
        with imageio.get_writer('{}/gif/{}.gif'.format(self.gif_dir, filename), mode='I', fps=5) as writer:
            for png_filename in sorted(os.listdir(self.gif_dir+'/png/'), key = lambda x: (len (x), x)):
                image = imageio.imread(self.gif_dir+'/png/'+png_filename)
                writer.append_data(image)

    ##################################################################
    # Plotting
    ##################################################################

    def build_plots(self, xp=[], belief=[], abs_sensor=None, abs_target=None, abs_particles=None, time_step=None, fig=None, ax=None):

        if len(self.abs_target_hist) < self.history_length:
            self.abs_target_hist = [abs_target] * self.history_length
            self.abs_sensor_hist = [abs_sensor] * self.history_length
        else:
            self.abs_target_hist.pop(0)
            self.abs_target_hist.append(abs_target)
            self.abs_sensor_hist.pop(0)
            self.abs_sensor_hist.append(abs_sensor)

        fig = plt.figure(figsize=(30, 6))
        plt.tight_layout()
        # Put space between plots
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        # Plot 1: Particle Plot (Polar)
        ax = fig.add_subplot(1, 5, 1, polar=True)
        grid_r, grid_theta = [],[]
        plot_r = [row[0] for row in belief]
        plot_theta = np.radians(np.array([row[1] for row in belief]))
        plot_x_theta = np.radians(xp[1])
        plot_x_r = xp[0]
        ax.plot(plot_theta, plot_r, 'ro')
        ax.plot(plot_x_theta, plot_x_r, 'bo')
        ax.set_ylim(-150,150)
        ax.set_title('iteration {}'.format(time_step), fontsize=16)


        # Plot 2: Particle Plot (Polar) with Interpolation
        ax = fig.add_subplot(1, 5, 2, polar=True)
        # Create grid values first via histogram.
        nbins = 10
        counts, xbins, ybins = np.histogram2d(plot_theta, plot_r, bins=nbins)
        # Make a meshgrid for theta, r values
        tm, rm = np.meshgrid(xbins[:-1], ybins[:-1])
        # Build contour plot
        ax.contourf(tm, rm, counts)
        # True position
        ax.plot(plot_x_theta, plot_x_r, 'bo')
        ax.set_ylim(-150,150)
        ax.set_title('Interpolated Belief'.format(time_step), fontsize=16)

        # Plot 3: Heatmap Plot (Cartesian)
        ax = fig.add_subplot(1, 5, 3)
        cart  = np.array(list(map(pol2cart, belief[:,0], np.radians(belief[:,1]))))
        x = cart[:,0]
        y = cart[:,1]
        xedges = np.arange(-150, 153, 3)
        yedges = np.arange(-150, 153, 3)
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        heatmap = gaussian_filter(heatmap, sigma=5)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im  = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='coolwarm')
        plt.colorbar(im)
        ax.set_xlim(-200,200)
        ax.set_ylim(-200,200)
        ax.set_title('Particle heatmap (relative to sensor)')

        # Plots 4 & 5: Absolute Particle/Sensor/Target Plot 
        # if abs_sensor is not None and abs_target is not None and abs_particles is not None:
        #     # particles/centroid coordinates
        #     particles_x, particles_y = pol2cart(abs_particles[:,0], np.radians(abs_particles[:,1]))
        #     centroid_x = np.mean(particles_x)
        #     centroid_y = np.mean(particles_y)
        #     centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)
        #     # target coordinates 
        #     target_x, target_y = pol2cart(abs_target[0], np.radians(abs_target[1]))
        #     # sensor coordinates
        #     sensor_x, sensor_y = pol2cart(abs_sensor[0], np.radians(abs_sensor[1]))

        #     # Plot 4: Absolute Polar coordinates 
        #     ax = fig.add_subplot(1, 5, 4, polar=True)
        #     ax.plot(np.radians(abs_particles[:,1]), abs_particles[:,0], 'ro', label='particles')
        #     ax.plot(centroid_theta, centroid_r, 'c*', label='centroid', markersize=12)
        #     ax.plot(np.radians(abs_sensor[1]), abs_sensor[0], 'gp', label='sensor', markersize=12)
        #     ax.plot(np.radians(abs_target[1]), abs_target[0], 'bX', label='target', markersize=12)
        #     ax.legend()
        #     ax.set_title('Absolute positions (polar)'.format(time_step), fontsize=16)

        #     # Plot 5: Absolute Cartesian coordinates
        #     ax = fig.add_subplot(1, 5, 5)
        #     xedges = np.arange(-100, 103, 3)
        #     yedges = np.arange(-100, 103, 3)
        #     heatmap, xedges, yedges = np.histogram2d(particles_x, particles_y, bins=(xedges, yedges))
        #     heatmap = gaussian_filter(heatmap, sigma=2)
        #     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #     im  = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='coolwarm')
        #     plt.colorbar(im)
        #     #ax.plot(particles_x, particles_y, 'ro', label='particles')
        #     ax.plot(centroid_x, centroid_y, 'c*', label='centroid', markersize=12)
        #     ax.plot(sensor_x, sensor_y, 'gp', label='sensor', markersize=12)
        #     ax.plot(target_x, target_y, 'mX', label='target', markersize=12)
        #     ax.legend()
        #     ax.set_xlim(-100,100)
        #     ax.set_ylim(-100,100)
        #     ax.set_title('Absolute positions (cartesian)'.format(time_step), fontsize=16)


        # Plots 4 & 5: Absolute Particle/Sensor/Target Plot
        # particles/centroid coordinates
        particles_x, particles_y = pol2cart(abs_particles[:,0], np.radians(abs_particles[:,1]))
        centroid_x = np.mean(particles_x)
        centroid_y = np.mean(particles_y)
        centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)
        sensor_r, sensor_theta, sensor_x, sensor_y  = [], [], [], []
        target_r, target_theta, target_x, target_y = [], [], [], []
        for i in range(5):
            sensor_r.append(self.abs_sensor_hist[10*(i+1)-1][0])
            sensor_theta.append(np.radians(self.abs_sensor_hist[10*(i+1)-1][1]))
            target_r.append(self.abs_target_hist[10*(i+1)-1][0])
            target_theta.append(np.radians(self.abs_target_hist[10*(i+1)-1][1]))
            sensor_x[i], sensor_y[i] = pol2cart(sensor_r, sensor_theta)
            target_x[i], target_y[i] = pol2cart(target_r, target_theta)

        # Plot 4: Absolute Polar coordinates 
        ax = fig.add_subplot(1, 5, 4, polar=True)
        ax.plot(np.radians(abs_particles[:,1]), abs_particles[:,0], 'ro', label='particles', alpha=0.5)
        ax.plot(centroid_theta, centroid_r, 'c*', label='centroid', markersize=12)
        ax.plot(sensor_theta[4], sensor_r[4], 'gp', label='sensor', markersize=12)
        ax.plot(target_theta[4], target_r[4], 'bX', label='target', markersize=12)
        for i in range(4):
            ax.plot(sensor_theta[i], sensor_r[i], 'gp', markersize=6, alpha=0.75)
            ax.plot(target_theta[i], target_r[i], 'bX', markersize=6, alpha=0.75)
        ax.legend()
        ax.set_title('Absolute positions (polar)'.format(time_step), fontsize=16)

        # Plot 5: Absolute Cartesian coordinates
        ax = fig.add_subplot(1, 5, 5)
        xedges = np.arange(-100, 103, 3)
        yedges = np.arange(-100, 103, 3)
        heatmap, xedges, yedges = np.histogram2d(particles_x, particles_y, bins=(xedges, yedges))
        heatmap = gaussian_filter(heatmap, sigma=2)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]        
        im  = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='coolwarm')
        plt.colorbar(im)
        #ax.plot(particles_x, particles_y, 'ro', label='particles', alpha=0.5)
        #ax.plot(sensor_x, sensor_y, 'gp', label='sensor', markersize=12)
        #ax.plot(target_x, target_y, 'mX', label='target', markersize=12)
        ax.plot(centroid_x, centroid_y, 'c*', label='centroid', markersize=12)
        ax.plot(sensor_x[4], sensor_y[4], 'gp', label='sensor', markersize=12)
        ax.plot(target_x[4], target_y[4], 'bX', label='target', markersize=12)
        for i in range(4):
            ax.plot(sensor_x[i], sensor_y[i], 'gp', markersize=6, alpha=0.55)
            ax.plot(target_x[i], target_y[i], 'bX', markersize=6, alpha=0.55)
        ax.legend()
        ax.set_xlim(-150,150)
        ax.set_ylim(-150,150)
        ax.set_title('Absolute positions (cartesian)'.format(time_step), fontsize=16)

        r_error, theta_error, heading_error, centroid_distance_error, rmse, mae  = tracking_error(abs_target, abs_particles)
        #print('r error = {:.0f}, theta error = {:.0f} deg, heading error = {:.0f} deg, centroid distance = {:.0f}, rmse = {:.0f}'.format(
        #    r_error, theta_error, heading_error, centroid_distance_error, rmse))

        png_filename = '{}/png/{}.png'.format(self.gif_dir, time_step)
        print('saving plots in {}'.format(png_filename))
        plt.savefig(png_filename)
        plt.close(fig)
        #plt.show()



##################################################################
# Logging
##################################################################
def write_header_log(config, method, global_start_time):

    config2log = {section: dict(config[section]) for section in config.sections()}

    #write output header
    run_dir = RUN_DIR
    if not os.path.isdir('{}/{}/'.format(RUN_DIR, method)):
        os.makedirs('{}/{}/'.format(RUN_DIR, method))
    header_filename = "{}/{}/{}_header.txt".format(RUN_DIR, method, global_start_time)
    with open(header_filename, "w") as f:
        f.write(json.dumps(config2log))

def read_header_log(filename):

    with open(filename) as f:
        config = json.load(f)
    return config

def particles_mean_belief(particles):
    particles_r = particles[:,0]
    particles_theta = np.radians(particles[:,1])
    particles_x, particles_y = pol2cart(particles_r, particles_theta)

    # centroid of particles x,y
    mean_x = np.mean(particles_x)
    mean_y = np.mean(particles_y)

    # centroid of particles r,theta
    mean_r, mean_theta = cart2pol(mean_x, mean_y)

    particles_heading = particles[:,2]
    particles_heading_rad = np.radians(particles_heading)
    mean_heading_rad = np.arctan2(np.mean(np.sin(particles_heading_rad)), np.mean(np.cos(particles_heading_rad)))
    mean_heading = np.degrees(mean_heading_rad)

    mean_spd = np.mean(particles[:,3])

    return particles_x, particles_y, mean_x, mean_y, mean_r, mean_theta, mean_heading, mean_spd

# calculate different tracking errors
def tracking_error(target, particles):

    target_r = target[0]
    target_theta = np.radians(target[1])
    target_heading = target[2]
    target_x, target_y = pol2cart(target_r, target_theta)

    particles_x, particles_y, mean_x, mean_y, mean_r, mean_theta, mean_heading, mean_spd = particles_mean_belief(particles)

    ## Error Measures
    #r_error = target_r - mean_r
    r_error = np.mean(target_r - particles[:,0])
    #theta_error = np.degrees(target_theta - mean_theta) # final error in degrees
    theta_error = np.mean(np.degrees(target_theta-np.radians(particles[:,1])))
    if theta_error > 360:
        theta_error = theta_error % 360
    #heading_diff = np.abs(target_heading - mean_heading) % 360
    heading_diff = np.abs(np.mean(target_heading - particles[:,2])) % 360 
    heading_error = heading_diff if heading_diff <= 180 else 360-heading_diff

    # centroid euclidean distance error x,y
    centroid_distance_error = np.sqrt((mean_x - target_x)**2 + (mean_y - target_y)**2)

    mae = np.mean(np.sqrt((particles_x-target_x)**2 + (particles_y - target_y)**2))

    # root mean square error
    rmse = np.sqrt(np.mean((particles_x - target_x)**2 + (particles_y - target_y)**2))

    return r_error, theta_error, heading_error, centroid_distance_error, rmse, mae

