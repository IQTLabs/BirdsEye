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
# Particle Filter helper functions
##################################################################
def permute_particle(particle):
    return np.hstack((particle[4:], particle[:4]))

def particle_swap(env):
    # 2000 x 8
    particles = np.copy(env.pf.particles)
    n_targets = env.state.n_targets
    state_dim = 4

    #print(env.pf.particles[0])
    # convert particles to cartesian
    for i in range(n_targets):
        x,y = pol2cart(particles[:,state_dim*i], np.radians(particles[:,(state_dim*i)+1]))
        particles[:,state_dim*i] = x
        particles[:,(state_dim*i)+1] = y

    swapped = True
    k = 0
    while swapped and k <10:
        k += 1
        #print('k-means run')
        swapped = False
        for i in range(len(particles)):
            original_particle = np.copy(particles[i])
            target_centroids = [np.mean(particles[:,state_dim*t:(state_dim*t)+2]) for t in range(n_targets)]
            distance = 0
            for t in range(n_targets):
                dif = particles[i,state_dim*t:(state_dim*t)+2] - target_centroids[t]
                distance += np.dot(dif,dif)

            permuted_particle = permute_particle(particles[i])
            particles[i] = permuted_particle
            permuted_target_centroids = [np.mean(particles[:,state_dim*t:(state_dim*t)+2]) for t in range(n_targets)]
            permuted_distance = 0
            for t in range(n_targets):
                dif = particles[i,state_dim*t:(state_dim*t)+2] - permuted_target_centroids[t]
                permuted_distance += np.dot(dif,dif)

            if distance < permuted_distance:
                particles[i] = original_particle
            else:
                swapped = True

    # convert particles to polar
    for i in range(n_targets):
        rho,phi = cart2pol(particles[:,state_dim*i], particles[:,(state_dim*i)+1])
        particles[:,state_dim*i] = rho
        particles[:,(state_dim*i)+1] = np.degrees(phi)

    env.pf.particles = particles
    #print(env.pf.particles[0])

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
    def build_multitarget_plots(self, env, time_step=None, fig=None, ax=None, centroid_distance_error=None, selected_plots=[1,2,3,4,5]):
        xp = env.state.target_state
        belief = env.pf.particles.reshape(len(env.pf.particles), env.state.n_targets, 4)
        abs_sensor = env.state.sensor_state
        abs_target = np.array(env.get_absolute_target())
        abs_particles = env.get_absolute_particles()

        # print('xp shape = ',xp.shape)
        # print('belief shape = ',belief.shape)
        # print('abs sensor shape = ',abs_sensor.shape)
        # print('abs_target shape = ',abs_target.shape)
        # print('abs_particles.shape = ',abs_particles.shape)

        textstr = '\n'.join((
        r'$\mathrm{Target 1 distance}=%.2f$' % (centroid_distance_error[0], ),
        r'$\mathrm{Target 2 distance}=%.2f$' % (centroid_distance_error[1], ),
        r'$\mathrm{Sum of distances}=%.2f$' % (np.sum(centroid_distance_error), )))


        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)



        if len(self.abs_target_hist) < self.history_length:
            self.abs_target_hist = [abs_target] * self.history_length
            self.abs_sensor_hist = [abs_sensor] * self.history_length
        else:
            self.abs_target_hist.pop(0)
            self.abs_target_hist.append(abs_target)
            self.abs_sensor_hist.pop(0)
            self.abs_sensor_hist.append(abs_sensor)

        fig = plt.figure(figsize=(30, 6), dpi=256)
        plt.tight_layout()
        # Put space between plots
        plt.subplots_adjust(wspace=0.7, hspace=0.2)

        color_array = [['salmon','darkred', 'red'],['lightskyblue','darkblue','blue']]

        plot_count = 0

        if 1 in selected_plots: 
            #####
            # Plot 1: Particle Plot (Polar)
            plot_count += 1
            ax = fig.add_subplot(1, len(selected_plots), plot_count, polar=True)

            for t in range(env.state.n_targets):
                # plot particles
                plot_theta = np.radians(belief[:,t,1]) # np.radians(np.array([row[1] for row in belief]))
                plot_r = belief[:,t,0] #[row[0] for row in belief]
                
                #ax.plot(plot_theta, plot_r, 'o', markeredgecolor='black', zorder=1)
                ax.plot(plot_theta, plot_r, 'o', color=color_array[t][0], markersize=4, markeredgecolor='black', label='particles', alpha=0.3, zorder=1)

                # plot targets
                plot_x_theta = np.radians(xp[t,1])
                plot_x_r = xp[t,0]
                #ax.plot(plot_x_theta, plot_x_r, 'X', markersize=10, zorder=2)
                #ax.plot(plot_x_theta, plot_x_r, 'X', color=color_array[t][2], markeredgecolor='white', label='target', markersize=12, zorder=2)
            ax.set_ylim(0,300)
            ax.set_title('iteration {}'.format(time_step), fontsize=16)
            # place a text box in upper left in axes coords
            #ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
            #####

        if 2 in selected_plots: 
            #####
            # Plot 2: Particle Plot (Polar) with Interpolation
            plot_count += 1
            ax = fig.add_subplot(1, len(selected_plots), plot_count, polar=True)

            for t in range(env.state.n_targets):
                # Create grid values first via histogram.
                nbins = 10
                plot_theta = np.radians(belief[:,t,1]) # np.radians(np.array([row[1] for row in belief]))
                plot_r = belief[:,t,0] #[row[0] for row in belief]
                counts, xbins, ybins = np.histogram2d(plot_theta, plot_r, bins=nbins)
                # Make a meshgrid for theta, r values
                tm, rm = np.meshgrid(xbins[:-1], ybins[:-1])
                # Build contour plot
                ax.contourf(tm, rm, counts)
                # True position
                plot_x_theta = np.radians(xp[t,1])
                plot_x_r = xp[t,0]
                ax.plot(plot_x_theta, plot_x_r, 'X')

            ax.set_ylim(0,300)
            ax.set_title('Interpolated Belief'.format(time_step), fontsize=16)
            #####

        if 3 in selected_plots: 
            #####
            # Plot 3: Heatmap Plot (Cartesian)
            plot_count += 1
            ax = fig.add_subplot(1, len(selected_plots), plot_count)
            #ax2 = fig.add_subplot(1, len(selected_plots)+1, plot_count+1)
            #axs = [ax, ax2]
            map_width = 600
            min_map = -1*int(map_width/2)
            max_map = int(map_width/2)
            cell_size = int((max_map - min_map)/max_map)
            cell_size = 2
            xedges = np.arange(min_map, max_map+cell_size, cell_size)
            yedges = np.arange(min_map, max_map+cell_size, cell_size)

            #### COMBINED; UNCOMMENT AFTER PAPER PLOT
            all_particles_x, all_particles_y = [],[]

            for t in range(env.state.n_targets):
                cart  = np.array(list(map(pol2cart, belief[:,t,0], np.radians(belief[:,t,1]))))
                x = cart[:,0]
                y = cart[:,1]
                all_particles_x.extend(x)
                all_particles_y.extend(y)

                #xedges = np.arange(min_map, max_map,cell_size)
                #yedges = np.arange(min_map, max_map,cell_size)
                #heatmap, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
                #heatmap = gaussian_filter(heatmap, sigma=8)
                #extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                #im  = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='jet')
                #plt.colorbar(im)

            heatmap, xedges, yedges = np.histogram2d(all_particles_x, all_particles_y, bins=(xedges, yedges))
            heatmap = gaussian_filter(heatmap, sigma=8)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im  = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='jet', interpolation='nearest')
            plt.colorbar(im)
            ax.set_xlim(min_map, max_map)
            ax.set_ylim(min_map, max_map)
            ax.set_title('Particle heatmap (relative to sensor)')

            # for t in range(env.state.n_targets):
            #     cart  = np.array(list(map(pol2cart, belief[:,t,0], np.radians(belief[:,t,1]))))
            #     x = cart[:,0]
            #     y = cart[:,1]

            #     heatmap, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
            #     heatmap = gaussian_filter(heatmap, sigma=8)
            #     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            #     im  = axs[t].imshow(heatmap.T, extent=extent, origin='lower', cmap='jet')
            #     #plt.colorbar(im)


            #     axs[t].set_xlim(min_map, max_map)
            #     axs[t].set_ylim(min_map, max_map)
            #     axs[t].set_title('Particle heatmap (relative to sensor)')


        # Plots 4 & 5: Absolute Particle/Sensor/Target Plot
        # particles/centroid coordinates
        for t in range(env.state.n_targets):
            particles_x, particles_y = pol2cart(abs_particles[:,0], np.radians(abs_particles[:,1]))
            centroid_x = np.mean(particles_x)
            centroid_y = np.mean(particles_y)
            centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)
            target_r, target_theta, target_x, target_y = [], [], [], []
            for i in range(5):
                target_r.append(self.abs_target_hist[10*(i+1)-1][0])
                target_theta.append(np.radians(self.abs_target_hist[10*(i+1)-1][1]))
            target_x, target_y = pol2cart(target_r, target_theta)
                # target_x.append(t_x)
                # target_y.append(t_y)

        sensor_r, sensor_theta, sensor_x, sensor_y  = [], [], [], []
        for i in range(5):
            sensor_r.append(self.abs_sensor_hist[10*(i+1)-1][0])
            sensor_theta.append(np.radians(self.abs_sensor_hist[10*(i+1)-1][1]))
        sensor_x, sensor_y = pol2cart(sensor_r, sensor_theta)
            # sensor_x.append(s_x)
            # sensor_y.append(s_y)

        if 4 in selected_plots: 
            # Plot 4: Absolute Polar coordinates
            plot_count += 1
            ax = fig.add_subplot(1, len(selected_plots), plot_count, polar=True)
            
            for t in range(env.state.n_targets):
                particles_x, particles_y = pol2cart(abs_particles[:,t,0], np.radians(abs_particles[:,t,1]))
                centroid_x = np.mean(particles_x)
                centroid_y = np.mean(particles_y)
                centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)
                target_r, target_theta, target_x, target_y = [], [], [], []

                for i in range(5):
                    target_r.append(self.abs_target_hist[10*(i+1)-1][env.state.n_targets-1-t][0])
                    target_theta.append(np.radians(self.abs_target_hist[10*(i+1)-1][env.state.n_targets-1-t][1]))
                target_x, target_y = pol2cart(target_r, target_theta)
                ax.plot(target_theta[4], target_r[4], 'X', color=color_array[t][2], markeredgecolor='white',label='target', markersize=12, zorder=4)

                ax.plot(np.radians(abs_particles[:,t,1]), abs_particles[:,t,0], 'o', color=color_array[t][0], markersize=4, markeredgecolor='black', label='particles', alpha=0.3, zorder=1)
                ax.plot(centroid_theta, centroid_r, '*', color=color_array[t][1],markeredgecolor='white', label='centroid', markersize=12, zorder=2)
                
                
                #for i in range(4): 
                #    ax.plot(target_theta[i], target_r[i], 'X', markersize=6, alpha=0.75, zorder=4)
            ax.plot(sensor_theta[4], sensor_r[4], 'p', color='limegreen', markeredgecolor='black', label='sensor', markersize=12, zorder=3)
            #for i in range(4):
            #    ax.plot(sensor_theta[i], sensor_r[i], 'bp', markersize=6, alpha=0.75, zorder=3)
            #ax.legend()
            ax.legend(loc='center left', bbox_to_anchor=(1.08,0.5), fancybox=True, shadow=True,)
            ax.set_ylim(0,300)
            ax.set_title('Absolute positions (polar)'.format(time_step), fontsize=16)

        if 5 in selected_plots: 
            # Plot 5: Absolute Cartesian coordinates
            plot_count += 1
            ax = fig.add_subplot(1, len(selected_plots), plot_count)
            xedges = np.arange(min_map, max_map, cell_size)
            yedges = np.arange(min_map, max_map, cell_size)
            heatmap_combined = None
            all_particles_x, all_particles_y = [],[]
            for t in range(env.state.n_targets):

                particles_x, particles_y = pol2cart(abs_particles[:,t,0], np.radians(abs_particles[:,t,1]))
                all_particles_x.extend(particles_x)
                all_particles_y.extend(particles_y)
                centroid_x = np.mean(particles_x)
                centroid_y = np.mean(particles_y)
                centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)
                target_r, target_theta, target_x, target_y = [], [], [], []
                for i in range(5):
                    target_r.append(self.abs_target_hist[10*(i+1)-1][t][0])
                    target_theta.append(np.radians(self.abs_target_hist[10*(i+1)-1][t][1]))
                target_x, target_y = pol2cart(target_r, target_theta)

                # heatmap, xedges, yedges = np.histogram2d(particles_x, particles_y, bins=(xedges, yedges))
                # heatmap = gaussian_filter(heatmap, sigma=16)
                # if heatmap_combined is None:
                #     heatmap_combined = heatmap
                # else:
                #     heatmap_combined += heatmap

                # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                # im  = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='jet')
                # plt.colorbar(im)


                #ax.plot(particles_x, particles_y, 'ro', label='particles', alpha=0.5)
                #ax.plot(sensor_x, sensor_y, 'gp', label='sensor', markersize=12)
                #ax.plot(target_x, target_y, 'mX', label='target', markersize=12)
                ax.plot(centroid_x, centroid_y, '*', label='centroid', markersize=12)
                
                ax.plot(target_x[4], target_y[4], 'X', label='target', markersize=12)
                #for i in range(4): 
                #    ax.plot(target_x[i], target_y[i], 'X', markersize=6, alpha=0.55)
            ax.plot(sensor_x[4], sensor_y[4], 'p', label='sensor', markersize=12)
            #for i in range(4):
            #    ax.plot(sensor_x[i], sensor_y[i], 'p', markersize=6, alpha=0.55)

            heatmap, xedges, yedges = np.histogram2d(all_particles_x, all_particles_y, bins=(xedges, yedges))
            heatmap = gaussian_filter(heatmap, sigma=8)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im  = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='jet', interpolation='nearest')
            plt.colorbar(im)

            #ax.legend()
            ax.legend(loc='center left', bbox_to_anchor=(1.2,0.5), fancybox=True, shadow=True,)
            ax.set_xlim(min_map, max_map)
            ax.set_ylim(min_map, max_map)
            ax.set_title('Absolute positions (cartesian)'.format(time_step), fontsize=16)

        #r_error, theta_error, heading_error, centroid_distance_error, rmse, mae  = tracking_error(env.state.target_state, env.pf.particles) #tracking_error(abs_target, abs_particles)
        #print('r error = {:.0f}, theta error = {:.0f} deg, heading error = {:.0f} deg, centroid distance = {:.0f}, rmse = {:.0f}'.format(
        #    r_error, theta_error, heading_error, centroid_distance_error, rmse))

        png_filename = '{}/png/{}.png'.format(self.gif_dir, time_step)
        print('saving plots in {}'.format(png_filename))
        plt.savefig(png_filename)
        plt.close(fig)
        #plt.show()

    def build_plots(self, xp=[], belief=[], abs_sensor=None, abs_target=None, abs_particles=None, time_step=None, fig=None, ax=None):
        print(belief.shape)
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

def particles_centroid_xy(particles):
    particles_r = particles[:,0]
    particles_theta = np.radians(particles[:,1])
    particles_x, particles_y = pol2cart(particles_r, particles_theta)

    # centroid of particles x,y
    mean_x = np.mean(particles_x)
    mean_y = np.mean(particles_y)

    return [mean_x, mean_y]

def angle_diff(angle): 

    diff =  angle % 360

    diff = (diff + 360) % 360

    if diff > 180:
        diff -= 360
    return diff

# calculate different tracking errors
def tracking_error(all_targets, all_particles):

    results = []
    n_targets = len(all_particles[0])//4

    # reorder targets to fit closest particles
    min_distance = None
    optimal_target_permutation = None
    for target_permutation in [[all_targets[0],all_targets[1]], [all_targets[1], all_targets[0]]]:
        distance = 0
        for t in range(n_targets):
            particle_centroid = np.array(particles_centroid_xy(all_particles[:,4*t:4*(t+1)]))
            target = np.array(pol2cart(target_permutation[t][0], np.radians(target_permutation[t][1])))
            distance += np.linalg.norm(particle_centroid-target)**2
            #print('target {}, distance = {}'.format(t,np.linalg.norm(particle_centroid-target) ))
        #print('sum distance = ',distance)
        if min_distance is None or distance < min_distance:
            min_distance = distance
            optimal_target_permutation = target_permutation

    #print('optimal min sum distance = ',min_distance)

    for t in range(n_targets):
        target = optimal_target_permutation[t]
        particles = all_particles[:,4*t:4*(t+1)]

        target_r = target[0]
        target_theta = np.radians(target[1])
        target_heading = target[2]
        target_x, target_y = pol2cart(target_r, target_theta)

        particles_x, particles_y, mean_x, mean_y, mean_r, mean_theta, mean_heading, mean_spd = particles_mean_belief(particles)

        ## Error Measures
        #r_error = target_r - mean_r
        r_error = np.mean(np.abs(target_r - particles[:,0]))
        #theta_error = np.degrees(target_theta - mean_theta) # final error in degrees
        #theta_error = np.mean(np.degrees(target_theta-np.radians(particles[:,1])))
        theta_error = np.mean(np.abs(angle_diff(target[1] - particles[:,1])))
        # if theta_error > 360:
        #     theta_error = theta_error % 360
        #heading_diff = np.abs(target_heading - mean_heading) % 360
        heading_diff = np.abs(np.mean(target_heading - particles[:,2])) % 360
        heading_error = heading_diff if heading_diff <= 180 else 360-heading_diff

        # centroid euclidean distance error x,y
        centroid_distance_error = np.sqrt((mean_x - target_x)**2 + (mean_y - target_y)**2)

        mae = np.mean(np.sqrt((particles_x-target_x)**2 + (particles_y - target_y)**2))

        # root mean square error
        rmse = np.sqrt(np.mean((particles_x - target_x)**2 + (particles_y - target_y)**2))

        results.append([r_error, theta_error, heading_error, centroid_distance_error, rmse, mae])
    results = np.array(results).T

    r_error = results[0]
    theta_error = results[1]
    heading_error = results[2]
    centroid_distance_error = results[3]
    rmse = results[4]
    mae = results[5]

    return r_error, theta_error, heading_error, centroid_distance_error, rmse, mae

