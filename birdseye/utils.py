# utils.py

import numpy as np
import json
import pandas as pd
from pathlib import Path
import imageio
from itertools import permutations
from collections import defaultdict
import configparser

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import math
import requests
from itertools import product
from io import BytesIO


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
# helper functions for lat/lon
##################################################################
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

class GPSVis:
    """
        modified from:
        https://github.com/tisljaricleo/GPS-visualization-Python
        MIT License
        Copyright (c) 2021 Leo Tišljarić

        Class for GPS data visualization using pre-downloaded OSM map in image format.
    """
    def __init__(self, position=None, map_path=None, bounds=None):
        """
        :param data_path: Path to file containing GPS records.
        :param map_path: Path to pre-downloaded OSM map in image format.
        :param bounds: Upper-left, and lower-right GPS points of the map (lat1, lon1, lat2, lon2).
        """
        #self.data_path = data_path
        self.position = position
        self.map_path = map_path
        self.bounds = bounds


        if self.map_path is not None and self.bounds is not None:
            self.img = self.create_image_from_map()
        elif self.position is not None:
            self.zoom = 17
            self.TILE_SIZE = 256
            distance = 200

            #coord = [45.598915, -122.679929]
            coord = self.position

            #x, y = point_to_pixels(coord[0],coord[1], zoom)
            lat_dist = distance/111111
            lon_dist = distance / (111111 * np.cos(np.radians(coord[0])))
            top, bot = coord[0] + lat_dist, coord[0] - lat_dist
            lef, rgt = coord[1] - lon_dist, coord[1] + lon_dist
            self.bounds = [top, lef, bot, rgt]

            self.img = self.create_image_from_position()
        self.get_ticks()


    def plot_map(self, axis1=None, output=None, save_as='resultMap.png'):
        """
        Method for plotting the map. You can choose to save it in file or to plot it.
        :param output: Type 'plot' to show the map or 'save' to save it. Default None
        :param save_as: Name and type of the resulting image.
        :return:
        """
        # create Fig and Axis if doesn't exist
        if axis1 is None:
            fig, axis1 = plt.subplots(figsize=(10, 13))

        # Plot background map
        axis1.imshow(np.flipud(self.img), alpha=0.6, origin='lower')# , extent=[-122.68450, -122.67505, 45.59494, 45.60311], aspect='auto')

        # Plot points
        # img_points = np.array(img_points)
        # axis1.plot(img_points[:,0],img_points[:,1], marker='.', markersize=12, color='red', linestyle='None')
        #axis1.plot(10, 10, marker='.', markersize=22, color='blue', linestyle='None')

        # Set axis dimensions, labels and tick marks
        axis1.set_xlim(0,int(self.width_meters))
        axis1.set_ylim(0,int(self.height_meters))
        axis1.set_xlabel('Longitude')
        axis1.set_ylabel('Latitude')
        axis1.set_xticks(np.linspace(0,int(self.width_meters),num=8))
        axis1.set_xticklabels(self.x_ticks)
        axis1.set_yticks(np.linspace(0,int(self.height_meters),num=8))
        axis1.set_yticklabels(self.y_ticks)
        axis1.grid()

        # Save or display
        if output == 'save':
            plt.savefig(save_as)
        elif output == 'plot':
            plt.show()

    def point_to_pixels(self, lat, lon, zoom):
            """convert gps coordinates to web mercator"""
            r = math.pow(2, zoom) * self.TILE_SIZE
            lat = math.radians(lat)

            x = int((lon + 180.0) / 360.0 * r)
            y = int((1.0 - math.log(math.tan(lat) + (1.0 / math.cos(lat))) / math.pi) / 2.0 * r)

            return x, y

    def create_image_from_position(self):
        URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png".format

        # print('top, bottom',top,bot)
        # print('left, right',lef,rgt)

        top, lef, bot, rgt = self.bounds

        x0, y0 = self.point_to_pixels(top, lef, self.zoom)
        x1, y1 = self.point_to_pixels(bot, rgt, self.zoom)

        x0_tile, y0_tile = int(x0 / self.TILE_SIZE), int(y0 / self.TILE_SIZE)
        x1_tile, y1_tile = math.ceil(x1 / self.TILE_SIZE), math.ceil(y1 / self.TILE_SIZE)

        assert (x1_tile - x0_tile) * (y1_tile - y0_tile) < 50, "That's too many tiles!"

        # full size image we'll add tiles to
        img = Image.new('RGB', (
            (x1_tile - x0_tile) * self.TILE_SIZE,
            (y1_tile - y0_tile) * self.TILE_SIZE))

        # loop through every tile inside our bounded box
        for x_tile, y_tile in product(range(x0_tile, x1_tile), range(y0_tile, y1_tile)):
            with requests.get(URL(x=x_tile, y=y_tile, z=self.zoom)) as resp:
                tile_img = Image.open(BytesIO(resp.content))
            # add each tile to the full size image
            img.paste(
                im=tile_img,
                box=((x_tile - x0_tile) * self.TILE_SIZE, (y_tile - y0_tile) * self.TILE_SIZE))

        x, y = x0_tile * self.TILE_SIZE, y0_tile * self.TILE_SIZE

        img = img.crop((
            int(x0-x),  # left
            int(y0-y),  # top
            int(x1-x),  # right
            int(y1-y))) # bottom

        self.width_meters = get_distance((self.bounds[0],self.bounds[1]),(self.bounds[0],self.bounds[3]))
        self.height_meters = get_distance((self.bounds[0],self.bounds[1]),(self.bounds[2],self.bounds[1]))
        img = img.resize((int(self.width_meters),int(self.height_meters)))

        return img

    def create_image_from_map(self):
        """
        Create the image that contains the original map and the GPS records.
        :param color: Color of the GPS records.
        :param width: Width of the drawn GPS records.
        :return:
        """

        img = Image.open(self.map_path, 'r')
        self.width_meters = get_distance((self.bounds[0],self.bounds[1]),(self.bounds[0],self.bounds[3]))
        self.height_meters = get_distance((self.bounds[0],self.bounds[1]),(self.bounds[2],self.bounds[1]))
        img = img.resize((int(self.width_meters),int(self.height_meters)))
        print('background image size (pixels) = ',img.size)


        # if self.data_path:
        #     data = pd.read_csv(self.data_path, names=['LATITUDE', 'LONGITUDE'], sep=',')
        #     gps_data = tuple(zip(data['LATITUDE'].values, data['LONGITUDE'].values))
        #     print(self.result_image.size[0], self.result_image.size[1])
        #     for d in gps_data:
        #         x1, y1 = self.scale_to_img(d, (self.result_image.size[0], self.result_image.size[1]))
        #         img_points.append((x1, y1))
        #         print(x1,y1)

        return img

    def scale_to_img(self, lat_lon, w_h):
        """
        Conversion from latitude and longitude to the image pixels.
        It is used for drawing the GPS records on the map image.
        :param lat_lon: GPS record to draw (lat1, lon1).
        :param w_h: Size of the map image (w, h).
        :return: Tuple containing x and y coordinates to draw on map image.
        """
        # https://gamedev.stackexchange.com/questions/33441/how-to-convert-a-number-from-one-min-max-set-to-another-min-max-set/33445
        lat_old = (self.bounds[2], self.bounds[0])
        new = (0, w_h[1])
        y = ((lat_lon[0] - lat_old[0]) * (new[1] - new[0]) / (lat_old[1] - lat_old[0])) + new[0]
        lon_old = (self.bounds[1], self.bounds[3])
        new = (0, w_h[0])
        x = ((lat_lon[1] - lon_old[0]) * (new[1] - new[0]) / (lon_old[1] - lon_old[0])) + new[0]
        # y must be reversed because the orientation of the image in the matplotlib.
        # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
        return int(x),int(y) # w_h[1] - int(y)

    def set_origin(self, lat_lon):

        self.origin = self.scale_to_img(lat_lon, (int(self.width_meters), int(self.height_meters)))

    def get_ticks(self):
        """
        Generates custom ticks based on the GPS coordinates of the map for the matplotlib output.
        :return:
        """
        self.x_ticks = map(
            lambda x: round(x, 6),
            np.linspace(self.bounds[1], self.bounds[3], num=8))
        self.y_ticks = map(
            lambda x: round(x, 6),
            np.linspace(self.bounds[2], self.bounds[0], num=8))
        # Ticks must be reversed because the orientation of the image in the matplotlib.
        # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
        self.y_ticks = list(self.y_ticks) #sorted(y_ticks, reverse=True)
        self.x_ticks = list(self.x_ticks)


##################################################################
# Saving Results
##################################################################
class Results:
    '''
    Results class for saving run results
    to file with common format.
    '''
    def __init__(self, method_name='', global_start_time='', num_iters=0, plotting=False, config=None, experiment_name=None):
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
        self.logdir = '{}/{}/{}_logs/'.format(RUN_DIR, method_name, global_start_time)

        Path(self.gif_dir+'/png/').mkdir(parents=True, exist_ok=True)
        Path(self.gif_dir+'/gif/').mkdir(parents=True, exist_ok=True)
        Path(self.logdir).mkdir(parents=True, exist_ok=True)
        self.col_names =['time', 'run_time', 'target_state', 'sensor_state',
                         'action', 'observation', 'reward', 'collisions', 'lost',
                         'r_err', 'theta_err', 'heading_err', 'centroid_err', 'rmse','mae','inference_times', 'pf_cov']

        self.pf_stats = defaultdict(list)
        self.abs_target_hist = []
        self.abs_sensor_hist = []
        self.target_hist = []
        self.sensor_hist = []
        self.history_length = 50
        self.time_step = 0
        self.texts = []
        self.openstreetmap = None
        self.transform = None

        if config:
            write_header_log(config, self.method_name, self.global_start_time)

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

    def live_plot(self, env, time_step=None, fig=None, ax=None, data=None, simulated=True, textstr=None):

        if self.openstreetmap is None and data['position'] is not None:
            self.openstreetmap = GPSVis(
              position = data['position']
              #map_path='map_delta_park.png',  # Path to map downloaded from the OSM.
              #bounds=(45.60311,-122.68450, 45.59494, -122.67505) # upper left, lower right
            )
            self.openstreetmap.set_origin(data['position'])
            self.transform = np.array([self.openstreetmap.origin[0], self.openstreetmap.origin[1]])


        self.time_step = time_step
        self.pf_stats['mean_hypothesis'].append(env.pf.mean_hypothesis)
        self.pf_stats['map_hypothesis'].append(env.pf.map_hypothesis)
        self.pf_stats['mean_state'].append(env.pf.mean_state)
        self.pf_stats['map_state'].append(env.pf.map_state)

        abs_sensor = env.state.sensor_state
        abs_particles = env.get_absolute_particles()
        self.sensor_hist.append(abs_sensor)

        ax.clear()
        if self.openstreetmap is not None:
            self.openstreetmap.plot_map(axis1=ax)
        ax.set_title('Time = {}'.format(time_step))

        color_array = [['salmon','darkred', 'red'],['lightskyblue','darkblue','blue']]
        lines = [] # https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.legend.html

        # Plot Particles
        for t in range(env.state.n_targets):
            particles_x, particles_y = pol2cart(abs_particles[:,t,0], np.radians(abs_particles[:,t,1]))
            if self.transform is not None:
                particles_x += self.transform[0]
                particles_y += self.transform[1]

            # centroid_x = np.mean(particles_x)
            # centroid_y = np.mean(particles_y)
            # centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)
            #ax.plot(centroid_theta, centroid_r, '*', color=color_array[t][1],markeredgecolor='white', label='centroid', markersize=12, zorder=2)
            line1, = ax.plot(particles_x, particles_y, 'o', color=color_array[t][0], markersize=4, markeredgecolor='black', label='particles', alpha=0.3, zorder=1)

            if t == 0:
                lines.extend([line1])
            else:
                lines.extend([])

        # Plot Sensor
        sensor_x, sensor_y = pol2cart(np.array(self.sensor_hist)[:,0], np.radians(np.array(self.sensor_hist)[:,1]))
        if self.transform is not None:
            sensor_x += self.transform[0]
            sensor_y += self.transform[1]
        if len(self.sensor_hist) > 1:
            ax.plot(sensor_x[:-1], sensor_y[:-1], linewidth=4.0, color='mediumorchid', zorder=3, markersize=12)
        line4, = ax.plot(sensor_x[-1], sensor_y[-1], 'H', color='mediumorchid', markeredgecolor='black', label='sensor', markersize=20, zorder=3)
        lines.extend([line4])

        # Legend
        ax.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5,-0.05), fancybox=True, shadow=True,ncol=2)

        # X/Y Limits
        if self.openstreetmap is None:
            map_width = 600
            min_map = -1*int(map_width/2)
            max_map = int(map_width/2)
            ax.set_xlim(min_map, max_map)
            ax.set_ylim(min_map, max_map)

        # Sidebar Text
        if textstr:
            last_mean_hyp = self.pf_stats['mean_hypothesis'][-1][0]
            last_map_hyp  = self.pf_stats['map_hypothesis'][-1][0]

            pfstats_str = ['Observed RSSI = {} dB\nML estimated RSSI =  {:.1f} dB\nMAP estimated RSSI = {:.1f} dB'.format(env.last_observation, last_mean_hyp, last_map_hyp)]
            if len(fig.texts) == 0:
                props = dict(boxstyle='round', facecolor='palegreen', alpha=0.5)
                text = fig.text(1.04, 0.75, textstr[0], transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
                props = dict(boxstyle='round', facecolor='paleturquoise', alpha=0.5)
                text = fig.text(1.04, 0.5, textstr[1], transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
                props = dict(boxstyle='round', facecolor='khaki', alpha=0.5)
                text = fig.text(1.04, 0.25, pfstats_str[0], transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
            else:
                fig.texts[0].set_text(textstr[0])
                fig.texts[1].set_text(textstr[1])
                fig.texts[2].set_text(pfstats_str[0])

    def build_multitarget_plots(self, env, time_step=None, fig=None, axs=None, centroid_distance_error=None, selected_plots=[1,2,3,4,5], simulated=True, textstr=None):
        xp = env.state.target_state
        belief = env.pf.particles.reshape(len(env.pf.particles), env.state.n_targets, 4)
        #print('sensor state = ',env.state.sensor_state)
        abs_sensor = env.state.sensor_state

        abs_particles = env.get_absolute_particles()

        if simulated:
            abs_target = np.array(env.get_absolute_target())
        else:
            abs_target = None
        # print('xp shape = ',xp.shape)
        # print('belief shape = ',belief.shape)
        # print('abs sensor shape = ',abs_sensor.shape)
        # print('abs_target shape = ',abs_target.shape)
        # print('abs_particles.shape = ',abs_particles.shape)

        # textstr = '\n'.join((
        # r'$\mathrm{Target 1 distance}=%.2f$' % (centroid_distance_error[0], ),
        # r'$\mathrm{Target 2 distance}=%.2f$' % (centroid_distance_error[1], ),
        # r'$\mathrm{Sum of distances}=%.2f$' % (np.sum(centroid_distance_error), )))


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

        if len(self.target_hist) == 150:
            self.target_hist = []
            self.sensor_hist = []
            self.rel_sensor_hist = []

        self.target_hist.append(abs_target)
        self.sensor_hist.append(abs_sensor)

        #fig = plt.figure(figsize=(10*len(selected_plots), 10), dpi=100)
        plt.tight_layout()
        # Put space between plots
        plt.subplots_adjust(wspace=0.7, hspace=0.2)

        color_array = [['salmon','darkred', 'red'],['lightskyblue','darkblue','blue']]

        plot_count = 0
        if axs is None:
            axs = {}

        map_width = 600
        min_map = -1*int(map_width/2)
        max_map = int(map_width/2)
        cell_size = int((max_map - min_map)/max_map)
        cell_size = 2
        xedges = np.arange(min_map, max_map+cell_size, cell_size)
        yedges = np.arange(min_map, max_map+cell_size, cell_size)

        if 1 in selected_plots:
            #####
            # Plot 1: Particle Plot (Polar)
            plot_count += 1
            if 1 not in axs:
                axs[1] = fig.add_subplot(1, len(selected_plots), plot_count, polar=True)
            ax = axs[1]
            ax.clear()

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
            #ax.set_title('iteration {}'.format(time_step), fontsize=16)
            # place a text box in upper left in axes coords
            #ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
            #####

        if 2 in selected_plots:
            #####
            # Plot 2: Particle Plot (Polar) with Interpolation
            plot_count += 1
            if 2 not in axs:
                axs[2] = fig.add_subplot(1, len(selected_plots), plot_count, polar=True)
            ax = axs[2]

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
            #ax.set_title('Interpolated Belief'.format(time_step), fontsize=16)
            #####

        if 3 in selected_plots:
            #####
            # Plot 3: Heatmap Plot (Cartesian)
            plot_count += 1
            if 3 not in axs:
                axs[3] = fig.add_subplot(1, len(selected_plots), plot_count)
            ax = axs[3]

            #ax2 = fig.add_subplot(1, len(selected_plots)+1, plot_count+1)
            #axs = [ax, ax2]


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
            #ax.set_title('Particle heatmap (relative to sensor)')

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
        # for t in range(env.state.n_targets):
        #     print(abs_particles.shape)
        #     print(abs_particles[:,0].shape)

        #     particles_x, particles_y = pol2cart(abs_particles[:,0], np.radians(abs_particles[:,1]))
        #     print(particles_x)
        #     asdf
        #     centroid_x = np.mean(particles_x)
        #     centroid_y = np.mean(particles_y)
        #     centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)
        #     target_r, target_theta, target_x, target_y = [], [], [], []
        #     for i in range(5):
        #         target_r.append(self.abs_target_hist[10*(i+1)-1][0])
        #         target_theta.append(np.radians(self.abs_target_hist[10*(i+1)-1][1]))
        #     target_x, target_y = pol2cart(target_r, target_theta)

        # sensor_r, sensor_theta, sensor_x, sensor_y  = [], [], [], []
        # for i in range(5):
        #     sensor_r.append(self.abs_sensor_hist[10*(i+1)-1][0])
        #     sensor_theta.append(np.radians(self.abs_sensor_hist[10*(i+1)-1][1]))
        # sensor_x, sensor_y = pol2cart(sensor_r, sensor_theta)


        if 4 in selected_plots:
            # Plot 4: Absolute Polar coordinates
            plot_count += 1
            if 4 not in axs:
                axs[4] = fig.add_subplot(1, len(selected_plots), plot_count, polar=True)
            ax = axs[4]
            ax.clear()

            lines = [] # https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.legend.html
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
                if len(self.target_hist) > 1:
                    ax.plot(np.radians(np.array(self.target_hist)[:-1,t,1]), np.array(self.target_hist)[:-1,t,0], linewidth=4.0, color='limegreen', zorder=3, markersize=12)

                line0, = ax.plot(target_theta[4], target_r[4], 'X', color='limegreen', markeredgecolor='black',label='targets', markersize=20, zorder=4)

                line1, = ax.plot(np.radians(abs_particles[:,t,1]), abs_particles[:,t,0], 'o', color=color_array[t][0], markersize=4, markeredgecolor='black', label='particles', alpha=0.3, zorder=1)
                #ax.plot(centroid_theta, centroid_r, '*', color=color_array[t][1],markeredgecolor='white', label='centroid', markersize=12, zorder=2)
                if t == 0:
                    lines.extend([line0,line1])
                else:
                    lines.extend([line0])


                #for i in range(4):
                #    ax.plot(target_theta[i], target_r[i], 'X', markersize=6, alpha=0.75, zorder=4)

            #line4, = ax.plot(sensor_theta[4], sensor_r[4], 'H', color='mediumorchid', markeredgecolor='black', label='sensor', markersize=20, zorder=3)

            if len(self.sensor_hist) > 1:
                ax.plot(np.radians(np.array(self.sensor_hist)[:-1,1]), np.array(self.sensor_hist)[:-1,0], linewidth=4.0, color='mediumorchid', zorder=3, markersize=12)

            line4, = ax.plot(np.radians(self.sensor_hist[-1][1]), self.sensor_hist[-1][0], 'H', color='mediumorchid', markeredgecolor='black', label='sensor', markersize=20, zorder=3)
            lines.extend([line4])
            #for i in range(4):
            #    ax.plot(sensor_theta[i], sensor_r[i], 'bp', markersize=6, alpha=0.75, zorder=3)
            #ax.legend()
            ax.legend(handles=lines, loc='center left', bbox_to_anchor=(1.08,0.5), fancybox=True, shadow=True,)
            ax.set_ylim(0,250)

            #ax.set_title('Absolute positions (polar)'.format(time_step), fontsize=16)

        if 5 in selected_plots:
            # Plot 5: Absolute Cartesian coordinates
            plot_count += 1
            if 5 not in axs:
                axs[5] = fig.add_subplot(1, len(selected_plots), plot_count)
            ax = axs[5]


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
            sensor_r, sensor_theta, sensor_x, sensor_y  = [], [], [], []
            for i in range(5):
                sensor_r.append(self.abs_sensor_hist[10*(i+1)-1][0])
                sensor_theta.append(np.radians(self.abs_sensor_hist[10*(i+1)-1][1]))
            sensor_x, sensor_y = pol2cart(sensor_r, sensor_theta)
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
            #ax.set_title('Absolute positions (cartesian)'.format(time_step), fontsize=16)

        if 6 in selected_plots:
            #####
            # Plot 1: Particle Plot (Polar)
            plot_count += 1
            if 6 not in axs:
                axs[6] = fig.add_subplot(1, len(selected_plots), plot_count)
            ax = axs[6]
            ax.clear()

            for t in range(env.state.n_targets):
                # plot particles
                plot_theta = np.radians(belief[:,t,1])
                plot_r = belief[:,t,0]
                particles_x, particles_y = pol2cart(belief[:,t,0], np.radians(belief[:,t,1]))
                ax.plot(particles_x, particles_y, 'o', color=color_array[t][0], markersize=4, markeredgecolor='black', label='particles', alpha=0.3, zorder=1)


                # plot targets
                plot_x_theta = np.radians(xp[t,1])
                plot_x_r = xp[t,0]
                #ax.plot(plot_x_theta, plot_x_r, 'X', markersize=10, zorder=2)
                #ax.plot(plot_x_theta, plot_x_r, 'X', color=color_array[t][2], markeredgecolor='white', label='target', markersize=12, zorder=2)
            ax.set_xlim(min_map, max_map)
            ax.set_ylim(min_map, max_map)

            sensor_x, sensor_y= pol2cart(self.sensor_hist[-1][0], np.radians(self.sensor_hist[-1][1]))

        if 7 in selected_plots:

            plot_count += 1
            if 7 not in axs:
                axs[7] = fig.add_subplot(1, len(selected_plots), plot_count)
            ax = axs[7]
            ax.clear()

            lines = [] # https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.legend.html
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
                target_x, target_y = pol2cart(np.array(self.target_hist)[:,t,0], np.radians(np.array(self.target_hist)[:,t,1]))

                if len(self.target_hist) > 1:
                    ax.plot(target_x[:-1], target_y[:-1], linewidth=4.0, color='limegreen', zorder=3, markersize=12)

                line0, = ax.plot(target_x[-1], target_y[-1], 'X', color='limegreen', markeredgecolor='black',label='targets', markersize=20, zorder=4)

                line1, = ax.plot(particles_x, particles_y, 'o', color=color_array[t][0], markersize=4, markeredgecolor='black', label='particles', alpha=0.3, zorder=1)
                #ax.plot(centroid_theta, centroid_r, '*', color=color_array[t][1],markeredgecolor='white', label='centroid', markersize=12, zorder=2)
                if t == 0:
                    lines.extend([line0,line1])
                else:
                    lines.extend([line0])


                #for i in range(4):
                #    ax.plot(target_theta[i], target_r[i], 'X', markersize=6, alpha=0.75, zorder=4)

            #line4, = ax.plot(sensor_theta[4], sensor_r[4], 'H', color='mediumorchid', markeredgecolor='black', label='sensor', markersize=20, zorder=3)

            sensor_x, sensor_y = pol2cart(np.array(self.sensor_hist)[:,0], np.radians(np.array(self.sensor_hist)[:,1]))
            if len(self.sensor_hist) > 1:
                ax.plot(sensor_x[:-1], sensor_y[:-1], linewidth=4.0, color='mediumorchid', zorder=3, markersize=12)

            line4, = ax.plot(sensor_x[-1], sensor_y[-1], 'H', color='mediumorchid', markeredgecolor='black', label='sensor', markersize=20, zorder=3)
            lines.extend([line4])
            #for i in range(4):
            #    ax.plot(sensor_theta[i], sensor_r[i], 'bp', markersize=6, alpha=0.75, zorder=3)
            #ax.legend()
            ax.legend(handles=lines, loc='center left', bbox_to_anchor=(1.08,0.5), fancybox=True, shadow=True,)

            ax.set_xlim(min_map, max_map)
            ax.set_ylim(min_map, max_map)
            #ax.set_title('Absolute positions (polar)'.format(time_step), fontsize=16)

        if 8 in selected_plots:

            plot_count += 1
            if 8 not in axs:
                axs[8] = fig.add_subplot(1, len(selected_plots), plot_count)
            ax = axs[8]
            ax.clear()

            lines = [] # https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.legend.html
            for t in range(env.state.n_targets):
                particles_x, particles_y = pol2cart(abs_particles[:,t,0], np.radians(abs_particles[:,t,1]))
                centroid_x = np.mean(particles_x)
                centroid_y = np.mean(particles_y)
                centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)


                line1, = ax.plot(particles_x, particles_y, 'o', color=color_array[t][0], markersize=4, markeredgecolor='black', label='particles', alpha=0.3, zorder=1)
                #ax.plot(centroid_theta, centroid_r, '*', color=color_array[t][1],markeredgecolor='white', label='centroid', markersize=12, zorder=2)
                if t == 0:
                    lines.extend([line1])
                else:
                    lines.extend([])


            sensor_x, sensor_y = pol2cart(np.array(self.sensor_hist)[:,0], np.radians(np.array(self.sensor_hist)[:,1]))
            if len(self.sensor_hist) > 1:
                ax.plot(sensor_x[:-1], sensor_y[:-1], linewidth=4.0, color='mediumorchid', zorder=3, markersize=12)

            line4, = ax.plot(sensor_x[-1], sensor_y[-1], 'H', color='mediumorchid', markeredgecolor='black', label='sensor', markersize=20, zorder=3)
            lines.extend([line4])
            #for i in range(4):
            #    ax.plot(sensor_theta[i], sensor_r[i], 'bp', markersize=6, alpha=0.75, zorder=3)
            #ax.legend()
            ax.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5,-0.05), fancybox=True, shadow=True,ncol=2)

            ax.set_xlim(min_map, max_map)
            ax.set_ylim(min_map, max_map)
            if textstr:
                props = dict(boxstyle='round', facecolor='palegreen', alpha=0.5)
                ax.text(1.04, 0.75, textstr[0], transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
                props = dict(boxstyle='round', facecolor='paleturquoise', alpha=0.5)
                ax.text(1.04, 0.5, textstr[1], transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
            #ax.set_title('Absolute positions (polar)'.format(time_step), fontsize=16)

        png_filename = '{}/png/{}.png'.format(self.gif_dir, time_step)
        #print('saving plots in {}'.format(png_filename))
        #plt.savefig(png_filename, bbox_inches='tight')
        #plt.close(fig)
        #plt.draw()
        #plt.show()
        return axs

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

    if type(config) == configparser.ConfigParser:
        config2log = {section: dict(config[section]) for section in config.sections()}
    else:
        config2log = dict(config)

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

    diff[diff > 180] -= 360
    return diff

# calculate different tracking errors
def tracking_error(all_targets, all_particles):

    results = []
    n_targets = len(all_particles[0])//4

    # reorder targets to fit closest particles
    min_distance = None
    optimal_target_permutation = None

    for idxs in list(permutations(range(n_targets))):
        target_permutation = all_targets[list(idxs)]

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

