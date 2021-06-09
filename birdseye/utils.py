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
        self.namefile = '{}/{}/{}_data.csv'.format(RUN_DIR, method_name, global_start_time)
        self.gif_dir = '{}/{}/{}'.format(RUN_DIR, method_name, global_start_time)

        Path(self.gif_dir+'/png/').mkdir(parents=True, exist_ok=True)
        Path(self.gif_dir+'/gif/').mkdir(parents=True, exist_ok=True)
        self.col_names =['time', 'run_time', 'target_state', 'sensor_state', 
                         'action', 'observation', 'reward', 'collisions', 'lost',
                         'r_err', 'theta_err', 'heading_err', 'centroid_err', 'rmse']

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
        with imageio.get_writer('{}/gif/{}.gif'.format(self.gif_dir, filename), mode='I') as writer:
            for png_filename in os.listdir(self.gif_dir+'/png/'):
                image = imageio.imread(self.gif_dir+'/png/'+png_filename)
                writer.append_data(image)

    ##################################################################
    # Plotting
    ##################################################################

    def build_plots(self, xp=[], belief=[], abs_sensor=None, abs_target=None, abs_particles=None, time_step=None, fig=None, ax=None):

        clear_output(wait=True)
        fig = plt.figure(figsize=(30, 6))
        plt.tight_layout()
        # Put space between plots
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        # Particle Plot (Polar)
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


        # Particle Plot (Polar) with Interpolation
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

        # Heatmap Plot (Cartesian)
        ax = fig.add_subplot(1, 5, 3)

        cart  = np.array(list(map(pol2cart, belief[:,0], np.radians(belief[:,1]))))
        x = cart[:,0]
        y = cart[:,1]
        xedges = np.arange(-100, 103, 3)
        yedges = np.arange(-100, 103, 3)
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        heatmap = gaussian_filter(heatmap, sigma=5)
        #heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        im  = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='coolwarm')
        plt.colorbar(im)
        ax.set_xlim(-200,200)
        ax.set_ylim(-200,200)
        ax.set_title('Particle heatmap (relative to sensor)')

        # Particle/Sensor Plot (absolute)
        if abs_sensor is not None and abs_target is not None and abs_particles is not None:

            particles_x, particles_y = pol2cart(abs_particles[:,0], np.radians(abs_particles[:,1]))
            centroid_x = np.mean(particles_x)
            centroid_y = np.mean(particles_y)
            centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)

            target_x, target_y = pol2cart(abs_target[0], np.radians(abs_target[1]))

            sensor_x, sensor_y = pol2cart(abs_sensor[0], np.radians(abs_sensor[1]))

            # Polar (absolute)
            ax = fig.add_subplot(1, 5, 4, polar=True)
            ax.plot(np.radians(abs_particles[:,1]), abs_particles[:,0], 'ro', label='particles')
            ax.plot(centroid_theta, centroid_r, 'c*', label='centroid', markersize=12)
            ax.plot(np.radians(abs_sensor[1]), abs_sensor[0], 'gp', label='sensor', markersize=12)
            ax.plot(np.radians(abs_target[1]), abs_target[0], 'bX', label='target', markersize=12)
            ax.legend()
            ax.set_title('Absolute positions (polar)'.format(time_step), fontsize=16)

            # Cartesian (absolute)
            ax = fig.add_subplot(1, 5, 5)
            
            xedges = np.arange(-100, 103, 3)
            yedges = np.arange(-100, 103, 3)
            heatmap, xedges, yedges = np.histogram2d(particles_x, particles_y, bins=(xedges, yedges))
            heatmap = gaussian_filter(heatmap, sigma=2)
            #heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            im  = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='coolwarm')
            plt.colorbar(im)

            #ax.plot(particles_x, particles_y, 'ro', label='particles')
            ax.plot(centroid_x, centroid_y, 'c*', label='centroid', markersize=12)
            ax.plot(sensor_x, sensor_y, 'gp', label='sensor', markersize=12)
            ax.plot(target_x, target_y, 'mX', label='target', markersize=12)
            ax.legend()
            ax.set_xlim(-100,100)
            ax.set_ylim(-100,100)
            ax.set_title('Absolute positions (cartesian)'.format(time_step), fontsize=16)

        
        r_error, theta_error, heading_error, centroid_distance_error, rmse  = tracking_error(abs_target, abs_particles)
        #print('r error = {:.0f}, theta error = {:.0f} deg, heading error = {:.0f} deg, centroid distance = {:.0f}, rmse = {:.0f}'.format(
        #    r_error, theta_error, heading_error, centroid_distance_error, rmse))
        
        plt.savefig('{}/png/{}.png'.format(self.gif_dir, time_step))
        #plt.show()



##################################################################
# Logging
##################################################################
def write_header_log(config, method, global_start_time):

    config2log = {section: dict(config[section]) for section in config.sections()}

    #write output header
    run_dir = RUN_DIR
    if not os.path.isdir('{}/{}/'.format(RUN_DIR, method)):
        os.mkdir('{}/{}/'.format(RUN_DIR, method))
    header_filename = "{}/{}/{}_header.txt".format(RUN_DIR, method, global_start_time)
    with open(header_filename, "w") as f:
        f.write(json.dumps(config2log))

def read_header_log(filename):

    with open(filename) as f:
        config = json.load(f)
    return config


##################################################################
# Plotting
##################################################################

def build_plots(xp=[], belief=[], abs_sensor=None, abs_target=None, abs_particles=None, time_step=None, fig=None, ax=None, global_start_time=None):

    clear_output(wait=True)
    fig = plt.figure(figsize=(30, 6))
    plt.tight_layout()
    # Put space between plots
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # Particle Plot (Polar)
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


    # Particle Plot (Polar) with Interpolation
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

    # Heatmap Plot (Cartesian)
    ax = fig.add_subplot(1, 5, 3)

    cart  = np.array(list(map(pol2cart, belief[:,0], np.radians(belief[:,1]))))
    x = cart[:,0]
    y = cart[:,1]
    xedges = np.arange(-100, 103, 3)
    yedges = np.arange(-100, 103, 3)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    heatmap = gaussian_filter(heatmap, sigma=5)
    #heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    im  = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='coolwarm')
    plt.colorbar(im)
    ax.set_xlim(-200,200)
    ax.set_ylim(-200,200)
    ax.set_title('Particle heatmap (relative to sensor)')

    # Particle/Sensor Plot (absolute)
    if abs_sensor is not None and abs_target is not None and abs_particles is not None:

        particles_x, particles_y = pol2cart(abs_particles[:,0], np.radians(abs_particles[:,1]))
        centroid_x = np.mean(particles_x)
        centroid_y = np.mean(particles_y)
        centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)

        target_x, target_y = pol2cart(abs_target[0], np.radians(abs_target[1]))

        sensor_x, sensor_y = pol2cart(abs_sensor[0], np.radians(abs_sensor[1]))

        # Polar (absolute)
        ax = fig.add_subplot(1, 5, 4, polar=True)
        ax.plot(np.radians(abs_particles[:,1]), abs_particles[:,0], 'ro', label='particles')
        ax.plot(centroid_theta, centroid_r, 'c*', label='centroid', markersize=12)
        ax.plot(np.radians(abs_sensor[1]), abs_sensor[0], 'gp', label='sensor', markersize=12)
        ax.plot(np.radians(abs_target[1]), abs_target[0], 'bX', label='target', markersize=12)
        ax.legend()
        ax.set_title('Absolute positions (polar)'.format(time_step), fontsize=16)

        # Cartesian (absolute)
        ax = fig.add_subplot(1, 5, 5)
        
        xedges = np.arange(-100, 103, 3)
        yedges = np.arange(-100, 103, 3)
        heatmap, xedges, yedges = np.histogram2d(particles_x, particles_y, bins=(xedges, yedges))
        heatmap = gaussian_filter(heatmap, sigma=2)
        #heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        im  = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='coolwarm')
        plt.colorbar(im)

        #ax.plot(particles_x, particles_y, 'ro', label='particles')
        ax.plot(centroid_x, centroid_y, 'c*', label='centroid', markersize=12)
        ax.plot(sensor_x, sensor_y, 'gp', label='sensor', markersize=12)
        ax.plot(target_x, target_y, 'mX', label='target', markersize=12)
        ax.legend()
        ax.set_xlim(-100,100)
        ax.set_ylim(-100,100)
        ax.set_title('Absolute positions (cartesian)'.format(time_step), fontsize=16)

    plt.show()
    r_error, theta_error, heading_error, centroid_distance_error, rmse  = tracking_error(abs_target, abs_particles)
    print('r error = {:.0f}, theta error = {:.0f} deg, heading error = {:.0f} deg, centroid distance = {:.0f}, rmse = {:.0f}'.format(
        r_error, theta_error, heading_error, centroid_distance_error, rmse))

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
    r_error = target_r - mean_r
    theta_error = np.degrees(target_theta - mean_theta) # final error in degrees
    if theta_error > 360:
        theta_error = theta_error % 360
    heading_diff = np.abs(target_heading - mean_heading) % 360
    heading_error = heading_diff if heading_diff <= 180 else 360-heading_diff

    # centroid euclidean distance error x,y
    centroid_distance_error = np.sqrt((mean_x - target_x)**2 + (mean_y - target_y)**2)

    # root mean square error
    rmse = np.sqrt(np.mean((particles_x - target_x)**2 + (particles_y - target_y)**2))

    return r_error, theta_error, heading_error, centroid_distance_error, rmse

