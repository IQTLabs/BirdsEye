"""
Particle Filter helper functions
"""
import argparse
import configparser
import json
import math
import os
import datetime
from collections import defaultdict
from io import BytesIO
from itertools import permutations
from itertools import product
from pathlib import Path
import pandas as pd
from pyparsing import col
import seaborn as sns


import imageio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

from .definitions import REPO_DIR
from .definitions import RUN_DIR


def targets_found(env, min_std_dev):
    std_dev = np.amax(
        env.get_particle_std_dev_cartesian(), axis=1
    )  # get maximum standard deviation axis for each target
    not_found = np.where(std_dev > min_std_dev)
    if len(not_found[0]) == 0:
        return True
    return False


def permute_particle(particle):
    return np.hstack((particle[4:], particle[:4]))


def particle_swap(env):
    # 2000 x 8
    particles = np.copy(env.pf.particles)
    n_targets = env.state.n_targets
    state_dim = 4

    # convert particles to cartesian
    for i in range(n_targets):
        x, y = pol2cart(
            particles[:, state_dim * i], np.radians(particles[:, (state_dim * i) + 1])
        )
        particles[:, state_dim * i] = x
        particles[:, (state_dim * i) + 1] = y

    swapped = True
    k = 0
    while swapped and k < 10:
        k += 1
        swapped = False
        for i in range(len(particles)):
            original_particle = np.copy(particles[i])
            target_centroids = [
                np.mean(particles[:, state_dim * t : (state_dim * t) + 2])
                for t in range(n_targets)
            ]
            distance = 0
            for t in range(n_targets):
                dif = (
                    particles[i, state_dim * t : (state_dim * t) + 2]
                    - target_centroids[t]
                )
                distance += np.dot(dif, dif)

            permuted_particle = permute_particle(particles[i])
            particles[i] = permuted_particle
            permuted_target_centroids = [
                np.mean(particles[:, state_dim * t : (state_dim * t) + 2])
                for t in range(n_targets)
            ]
            permuted_distance = 0
            for t in range(n_targets):
                dif = (
                    particles[i, state_dim * t : (state_dim * t) + 2]
                    - permuted_target_centroids[t]
                )
                permuted_distance += np.dot(dif, dif)

            if distance < permuted_distance:
                particles[i] = original_particle
            else:
                swapped = True

    # convert particles to polar
    for i in range(n_targets):
        rho, phi = cart2pol(
            particles[:, state_dim * i], particles[:, (state_dim * i) + 1]
        )
        particles[:, state_dim * i] = rho
        particles[:, (state_dim * i) + 1] = np.degrees(phi)

    env.pf.particles = particles


def circ_tangents(point, center, radius):
    px, py = point
    cx, cy = center

    b = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    if radius >= b:
        ##print(f"Warning: No tangents are possible.  radius >= distance to circle center ({radius} >= {b})")
        return None
    th = np.arccos(radius / b)
    d = np.arctan2(py - cy, px - cx)
    d1 = d + th
    d2 = d - th

    tangents = [
        [cx + radius * np.cos(d1), cy + radius * np.sin(d1)],
        [cx + radius * np.cos(d2), cy + radius * np.sin(d2)],
    ]
    v = np.array(point) - np.array(center)
    v /= np.linalg.norm(v)
    intersection = np.array(center) + radius * v
    tangents.append(list(intersection))
    return np.array(tangents)


def pol2cart(rho, phi):
    """
    Transform polar to cartesian
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def cart2pol(x, y):
    """
    Transform cartesian to polar
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def get_distance(coord1, coord2):
    """
    Get the distance between two coordinates
    """
    if (coord1 is None) or (coord2 is None):
        return None

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

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance * (1e3)


def get_heading(coord1, coord2):
    """
    Get the heading of two coordinates
    """
    if (coord1 is None) or (coord2 is None):
        return None

    lat1, long1 = coord1
    lat2, long2 = coord2
    dLon = long2 - long1
    x = np.cos(np.radians(lat2)) * np.sin(np.radians(dLon))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(
        np.radians(lat1)
    ) * np.cos(np.radians(lat2)) * np.cos(np.radians(dLon))
    brng = np.arctan2(x, y)
    brng = np.degrees(brng)

    return -brng + 90


def is_float(element):
    """
    Check if an element is a float or not
    """
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


class GPSVis:
    """
    modified from:
    https://github.com/tisljaricleo/GPS-visualization-Python
    MIT License
    Copyright (c) 2021 Leo Tišljarić

    Class for GPS data visualization using pre-downloaded OSM map in image format.
    """

    def __init__(self, position=None, map_path=None, bounds=None, distance=200):
        """
        :param data_path: Path to file containing GPS records.
        :param map_path: Path to pre-downloaded OSM map in image format.
        :param bounds: Upper-left, and lower-right GPS points of the map (lat1, lon1, lat2, lon2).
        """
        self.position = position
        self.map_path = map_path
        self.bounds = bounds
        self.origin = (0, 0)

        if self.map_path is not None and self.bounds is not None:
            self.img = self.create_image_from_map()
        elif self.position is not None:
            if distance <= 200:
                self.zoom = 18
            elif distance <= 1000:
                self.zoom = 16
            elif distance <= 2000:
                self.zoom = 14
            else:
                self.zoom = 12
            self.TILE_SIZE = 256
            distance = distance

            coord = self.position

            lat_dist = distance / 111111
            lon_dist = distance / (111111 * np.cos(np.radians(coord[0])))
            top, bot = coord[0] + lat_dist, coord[0] - lat_dist
            lef, rgt = coord[1] - lon_dist, coord[1] + lon_dist
            self.bounds = [top, lef, bot, rgt]

            self.img = self.create_image_from_position()
        # TODO if else self.width_meters and self.height_meters are undefined
        self.get_ticks()
        self.cell_size = 1
        self.xedges = np.arange(0, self.width_meters + self.cell_size, self.cell_size)
        self.yedges = np.arange(0, self.height_meters + self.cell_size, self.cell_size)

    def plot_map(self, axis1=None, output=None, save_as="resultMap.png"):
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
        axis1.imshow(np.flipud(self.img), alpha=0.7, origin="lower")

        # Set axis dimensions, labels and tick marks
        axis1.set_xlim(0, int(self.width_meters))
        axis1.set_ylim(0, int(self.height_meters))
        axis1.set_xlabel("Longitude")
        axis1.set_ylabel("Latitude")
        axis1.set_xticks(np.linspace(0, int(self.width_meters), num=8))
        axis1.set_xticklabels(self.x_ticks, rotation=30, ha="center")
        axis1.set_yticks(np.linspace(0, int(self.height_meters), num=8))
        axis1.set_yticklabels(self.y_ticks)
        axis1.grid()

        # Save or display
        if output == "save":
            plt.savefig(save_as)
        elif output == "plot":
            plt.show()

    def point_to_pixels(self, lat, lon, zoom):
        """convert gps coordinates to web mercator"""
        r = math.pow(2, zoom) * self.TILE_SIZE
        lat = math.radians(lat)

        x = int((lon + 180.0) / 360.0 * r)
        y = int(
            (1.0 - math.log(math.tan(lat) + (1.0 / math.cos(lat))) / math.pi) / 2.0 * r
        )

        return x, y

    def create_image_from_position(self):
        URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png".format

        top, lef, bot, rgt = self.bounds

        x0, y0 = self.point_to_pixels(top, lef, self.zoom)
        x1, y1 = self.point_to_pixels(bot, rgt, self.zoom)

        x0_tile, y0_tile = int(x0 / self.TILE_SIZE), int(y0 / self.TILE_SIZE)
        x1_tile, y1_tile = math.ceil(x1 / self.TILE_SIZE), math.ceil(
            y1 / self.TILE_SIZE
        )

        assert (x1_tile - x0_tile) * (y1_tile - y0_tile) < 50, "That's too many tiles!"

        # full size image we'll add tiles to
        img = Image.new(
            "RGB",
            (
                (x1_tile - x0_tile) * self.TILE_SIZE,
                (y1_tile - y0_tile) * self.TILE_SIZE,
            ),
        )

        # loop through every tile inside our bounded box
        for x_tile, y_tile in product(range(x0_tile, x1_tile), range(y0_tile, y1_tile)):
            try:
                with requests.get(
                    URL(x=x_tile, y=y_tile, z=self.zoom),
                    headers={"User-Agent": "BirdsEye/0.1.1"},
                ) as resp:
                    tile_img = Image.open(BytesIO(resp.content))
            except Exception as e:
                tile_img = Image.open(f"{REPO_DIR}/data/0.png")
            # add each tile to the full size image
            img.paste(
                im=tile_img,
                box=(
                    (x_tile - x0_tile) * self.TILE_SIZE,
                    (y_tile - y0_tile) * self.TILE_SIZE,
                ),
            )

        x, y = x0_tile * self.TILE_SIZE, y0_tile * self.TILE_SIZE

        img = img.crop(
            (int(x0 - x), int(y0 - y), int(x1 - x), int(y1 - y))  # left  # top  # right
        )  # bottom

        self.width_meters = get_distance(
            (self.bounds[0], self.bounds[1]), (self.bounds[0], self.bounds[3])
        )
        self.height_meters = get_distance(
            (self.bounds[0], self.bounds[1]), (self.bounds[2], self.bounds[1])
        )
        img = img.resize((int(self.width_meters), int(self.height_meters)))

        return img

    def create_image_from_map(self):
        """
        Create the image that contains the original map and the GPS records.
        :param color: Color of the GPS records.
        :param width: Width of the drawn GPS records.
        :return:
        """

        img = Image.open(self.map_path, "r")
        self.width_meters = get_distance(
            (self.bounds[0], self.bounds[1]), (self.bounds[0], self.bounds[3])
        )
        self.height_meters = get_distance(
            (self.bounds[0], self.bounds[1]), (self.bounds[2], self.bounds[1])
        )
        img = img.resize((int(self.width_meters), int(self.height_meters)))
        print("background image size (pixels) = ", img.size)

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
        y = (
            (lat_lon[0] - lat_old[0]) * (new[1] - new[0]) / (lat_old[1] - lat_old[0])
        ) + new[0]
        lon_old = (self.bounds[1], self.bounds[3])
        new = (0, w_h[0])
        x = (
            (lat_lon[1] - lon_old[0]) * (new[1] - new[0]) / (lon_old[1] - lon_old[0])
        ) + new[0]
        # y must be reversed because the orientation of the image in the matplotlib.
        # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
        return [int(x), int(y)]

    def set_origin(self, lat_lon):
        self.origin = self.scale_to_img(
            lat_lon, (int(self.width_meters), int(self.height_meters))
        )

    def get_ticks(self):
        """
        Generates custom ticks based on the GPS coordinates of the map for the matplotlib output.
        :return:
        """
        self.x_ticks = map(
            lambda x: round(x, 4), np.linspace(self.bounds[1], self.bounds[3], num=8)
        )
        self.y_ticks = map(
            lambda x: round(x, 4), np.linspace(self.bounds[2], self.bounds[0], num=8)
        )
        # Ticks must be reversed because the orientation of the image in the matplotlib.
        # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
        self.y_ticks = list(self.y_ticks)  # sorted(y_ticks, reverse=True)
        self.x_ticks = list(self.x_ticks)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ResultsReader:
    """
    ResultsReader class for loading run results
    """

    def __init__(
        self,
        experiment_name="",
    ):
        self.parent_logs_dir = f"{RUN_DIR}/{experiment_name}/"
        self.log_dirs = [
            f"{self.parent_logs_dir}{d}"
            for d in os.listdir(self.parent_logs_dir)
            if d.endswith("_logs")
        ]
        self.log_data = {}
        self.log_config = {}
        for d in self.log_dirs:
            log_file = f"{d}/data.log"
            if os.path.exists(log_file):
                self.log_data[d] = self.load_log(f"{d}/data.log")
            config_file = f"{d}/config.log"
            self.log_config[d] = read_header_log(config_file)

            # fix missing target_speed
            if "target_speed" not in self.log_config[d]:
                self.log_config[d]["target_speed"] = 0.5

    def average_plantime(self):
        """
        Get average planning time
        """
        plan_time = []
        for run, log_data in self.log_data.items():
            times = [data["plan_time"] for data in log_data]
            plan_time.append(np.mean(times))
        avg_plan_time = np.mean(plan_time)
        return avg_plan_time

    def average_std_dev(self):
        """
        Get average of the max standard deviation dimension of the particle distributions
        """
        std_dev_all = []
        std_dev_success = []
        for run, log_data in self.log_data.items():
            std_dev = np.max(log_data[-1]["std_dev_cartesian"], axis=1)
            if len(log_data) < 400:
                std_dev_success.extend(std_dev)
            std_dev_all.extend(std_dev)
        avg_std_dev_all = np.mean(std_dev_all)
        avg_std_dev_success = np.mean(std_dev_success)
        return avg_std_dev_success, avg_std_dev_all

    def std_dev_plot(self, ax=None, color=None, facecolor=None):
        """
        Get average of the max standard deviation dimension of the particle distributions
        """
        std_dev_all = []
        std_dev_success = []
        for run, log_data in self.log_data.items():
            std_dev = [
                np.mean(np.max(np.array(d["std_dev_cartesian"]), axis=1))
                for d in log_data
            ]
            std_dev_all.append(std_dev)
        # if ax is None:
        #     fig = plt.figure()
        #     ax = fig.subplots()
        # if color is None:
        #     color="blue"
        # for sd in std_dev_all:
        #     ax.scatter(range(len(sd)), sd, s=2, color=color, marker=".", alpha=0.25)
        # plt.show()

        df = pd.DataFrame(std_dev_all)
        if df.shape[1] < 400:
            df = df.reindex(columns=list(range(400)))
        df = df[:][list(range(0, 400, 20))]
        df.boxplot(
            ax=ax,
            notch=True,
            patch_artist=True,
            boxprops=dict(facecolor=facecolor, color=color, alpha=0.75),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color),
            medianprops=dict(color=color),
            # color=color,
            showfliers=False,
        )

        # if experiment_name is not None:
        #     plt.title(experiment_name)
        # plt.show()

    # df.boxplot()

    def average_rmse(self):
        """
        Get average rmse for successful and all runs
        """
        rmse_success = []
        rmse_all = []
        for run, log_data in self.log_data.items():
            rmse = np.sqrt(
                np.mean(np.array(log_data[-1]["centroid_distance_err"]) ** 2)
            )
            if len(log_data) < 400:
                rmse_success.append(rmse)
            rmse_all.append(rmse)

        avg_rmse_success = np.mean(rmse_success)
        avg_rmse_all = np.mean(rmse_all)
        return avg_rmse_success, avg_rmse_all

    def rmse_plot(self, ax=None, color=None, facecolor=None):
        """
        Get average of the max standard deviation dimension of the particle distributions
        """
        rmse_all = []
        for run, log_data in self.log_data.items():
            rmse = [
                np.sqrt(np.mean(np.array(d["centroid_distance_err"]) ** 2))
                for d in log_data
            ]
            rmse_all.append(rmse)
        # if ax is None:
        #     fig = plt.figure()
        #     ax = fig.subplots()
        # if color is None:
        #     color="blue"
        # for sd in std_dev_all:
        #     ax.scatter(range(len(sd)), sd, s=2, color=color, marker=".", alpha=0.25)
        # plt.show()

        df = pd.DataFrame(rmse_all)
        if df.shape[1] < 400:
            df = df.reindex(columns=list(range(400)))
        df = df[:][list(range(0, 400, 20))]
        df.boxplot(
            ax=ax,
            notch=True,
            patch_artist=True,
            boxprops=dict(facecolor=facecolor, color=color, alpha=0.75),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color),
            medianprops=dict(color=color),
            # color=color,
            showfliers=False,
        )
        # if experiment_name is not None:
        #     plt.title(experiment_name)
        # plt.show()

    def rmse_plot2(self):
        """
        Get average rmse for successful and all runs
        """
        rmse_success = []
        rmse_all = []
        for run, log_data in self.log_data.items():
            rmse = np.sqrt(
                np.mean(np.array(log_data[-1]["centroid_distance_err"]) ** 2)
            )
            if len(log_data) < 400:
                rmse_success.append(rmse)
            rmse_all.append(rmse)

        avg_rmse_success = np.mean(rmse_success)
        avg_rmse_all = np.mean(rmse_all)
        return avg_rmse_success, avg_rmse_all

    def localization_probability(self):
        """
        Get the probability of successful localizations from experiment run data.
        """
        success_localize = 0
        for run, log_data in self.log_data.items():
            if len(log_data) < 400:
                success_localize += 1
        success_localize_prob = success_localize / len(self.log_data)
        return success_localize_prob

    def average_localization_time(self):
        """
        Get the average run time of successful localization runs.
        """
        success_localize = 0
        average_localize_time = 0
        for run, log_data in self.log_data.items():
            if len(log_data) < 400:
                success_localize += 1
                average_localize_time += len(log_data)
        average_localize_time /= success_localize
        return average_localize_time

    def localization_histogram(self, ax=None, color=None):
        """
        Get the average run time of successful localization runs.
        """

        runtime = []
        for run, log_data in self.log_data.items():
            runtime.append(len(log_data))

        # ax.hist(runtime, color=color, bins=400)
        sns.histplot(
            data=runtime, kde=True, ax=ax, color=color, bins=np.arange(0, 420, 20)
        )

    def load_log(self, log_file):
        """
        Load json log file
        """
        data = []
        with open(
            log_file,
            "r",
            encoding="UTF-8",
        ) as infile:
            for line in infile:
                data.append(json.loads(line))
        return data


class Results:
    """
    Results class for saving run results
    to file with common format.
    """

    def __init__(
        self,
        experiment_name="",
        global_start_time="",
        num_iters=0,
        plotting=False,
        config={},
        enable_heatmap=False,
        enable_gps_plot=False,
        class_map={},
    ):
        self.num_iters = num_iters
        self.experiment_name = experiment_name
        self.global_start_time = global_start_time
        self.plotting = plotting
        self.enable_heatmap = enable_heatmap
        self.enable_gps_plot = enable_gps_plot
        self.class_map = class_map
        if not isinstance(self.plotting, bool):
            if self.plotting in ("true", "True"):
                self.plotting = True
            else:
                self.plotting = False
        if type(config) == argparse.Namespace:
            config = vars(config)
        self.native_plot = config.get("native_plot", "false").lower()
        self.plot_every_n = int(config.get("plot_every_n", 1))
        self.make_gif = config.get("make_gif", "false").lower()
        self.namefile = f"{RUN_DIR}/{experiment_name}/{global_start_time}_data.csv"
        self.logdir = f"{RUN_DIR}/{experiment_name}/{global_start_time}_logs/"
        Path(self.logdir).mkdir(parents=True, exist_ok=True)
        self.plot_dir = config.get("plot_dir", self.logdir)
        Path(self.plot_dir + "/png/").mkdir(parents=True, exist_ok=True)
        if self.make_gif == "true":
            Path(self.plot_dir + "/gif/").mkdir(parents=True, exist_ok=True)

        self.col_names = [
            "time",
            "run_time",
            "target_state",
            "sensor_state",
            "action",
            "observation",
            "reward",
            "collisions",
            "lost",
            "r_err",
            "theta_err",
            "heading_err",
            "centroid_err",
            "rmse",
            "mae",
            "inference_times",
            "pf_cov",
        ]

        self.pf_stats = defaultdict(list)
        self.abs_target_hist = []
        self.abs_sensor_hist = []
        self.target_hist = []
        self.sensor_hist = []
        self.sensor_gps_hist = []
        self.target_gps_hist = {}
        self.history_length = 10
        self.time_step = 0
        self.texts = []
        self.openstreetmap = None
        self.transform = None
        self.expected_target_rssi = None
        self.target_only_map = False

        if config:
            write_config_log(config, self.logdir)

    def data_to_npy(self, array, label, timestep):
        """
        Save npy array to file
        """
        Path(f"{self.logdir}/{label}/").mkdir(parents=True, exist_ok=True)
        np.save(
            f"{self.logdir}/{label}/{timestep}.npy",
            array,
        )

    def data_to_json(self, data):
        """
        Save data dict to log
        """
        with open(
            f"{self.logdir}/data.log",
            "a",
            encoding="UTF-8",
        ) as outfile:
            json.dump(data, outfile, cls=NumpyEncoder)
            outfile.write("\n")

    def write_dataframe(self, run_data):
        """
        Save dataframe to CSV file
        """
        if os.path.isfile(self.namefile):
            print("Updating file {}".format(self.namefile))
        else:
            print("Saving file to {}".format(self.namefile))
        df = pd.DataFrame(run_data, columns=self.col_names)
        df.to_csv(self.namefile)

    def save_gif(self, run, sub_run=None):
        filename = run if sub_run is None else "{}_{}".format(run, sub_run)
        # Build GIF
        with imageio.get_writer(
            "{}/gif/{}.gif".format(self.plot_dir, filename),
            mode="I",
            duration=int(1000 * 1 / 5),
        ) as writer:
            for png_filename in sorted(
                os.listdir(self.plot_dir + "/png/"), key=lambda x: (len(x), x)
            ):
                image = imageio.v2.imread(self.plot_dir + "/png/" + png_filename)
                writer.append_data(image)

    def live_plot(
        self,
        env,
        time_step=None,
        fig=None,
        ax=None,
        data=None,
        sidebar=False,
        separable=False,
        map_distance=200,
    ):
        """
        Create a live plot
        """

        lines = (
            []
        )  # https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.legend.html
        legend_elements = []
        separable_color_array = [
            ["deepskyblue", "blue"],
            ["pink", "red"],
            ["wheat", "orange"],
            ["lightgreen", "green"],
        ]
        color_array = [
            ["salmon", "darkred", "red"],
            ["lightskyblue", "darkblue", "blue"],
        ]
        sensor_color = "green"

        # Target only openstreetmap
        if self.openstreetmap is None and data.get("targets", None):
            self.target_only_map = True
            self.openstreetmap = GPSVis(
                position=data["targets"][list(data["targets"])[0]]["position"],
                distance=map_distance,
            )
            self.openstreetmap.set_origin(
                data["targets"][list(data["targets"])[0]]["position"]
            )
            self.transform = np.array(
                [self.openstreetmap.origin[0], self.openstreetmap.origin[1]]
            )

        # Openstreetmap
        if (
            (self.openstreetmap is None or self.target_only_map)
            and data.get("position", None) is not None
        ):
            self.target_only_map = False
            self.target_gps_hist = {}
            self.openstreetmap = GPSVis(
                position=data["position"], distance=map_distance
            )
            self.openstreetmap.set_origin(data["position"])
            self.transform = np.array(
                [self.openstreetmap.origin[0], self.openstreetmap.origin[1]]
            )

        self.time_step = time_step

        # Particle filter statistics
        self.pf_stats["mean_hypothesis"].append(
            env.pf.mean_hypothesis if hasattr(env.pf, "mean_hypothesis") else [None]
        )
        self.pf_stats["map_hypothesis"].append(
            env.pf.map_hypothesis if hasattr(env.pf, "map_hypothesis") else [None]
        )
        self.pf_stats["mean_state"].append(
            env.pf.mean_state if hasattr(env.pf, "mean_state") else [None]
        )
        self.pf_stats["map_state"].append(
            env.pf.map_state if hasattr(env.pf, "map_state") else [None]
        )

        # Sensor state history (from internal state)
        abs_sensor = env.state.sensor_state
        self.sensor_hist.append(abs_sensor)

        # Sensor state history (from gps)
        if self.openstreetmap and data.get("position", None) is not None:
            self.sensor_gps_hist.append(
                self.openstreetmap.scale_to_img(
                    data["position"],
                    (self.openstreetmap.width_meters, self.openstreetmap.height_meters),
                )
            )

        # Particle states
        abs_particles = env.get_absolute_particles()

        # Target state history (from internal state)
        if env.simulated:
            self.target_hist.append(env.get_absolute_target())

        # Target state history (from gps)
        if self.openstreetmap and data.get("target_gps", None) is not None:
            for t in data["targets"]:
                if t not in self.target_gps_hist:
                    self.target_gps_hist[t] = []
                self.target_gps_hist[t].append(
                    self.openstreetmap.scale_to_img(
                        data["targets"][t]["position"],
                        (
                            self.openstreetmap.width_meters,
                            self.openstreetmap.height_meters,
                        ),
                    )
                )
                # print(
                #     f"\n{t}:{data['targets'][t]['position']}, {self.target_gps_hist[t][-1]}\n"
                # )

        # Target state history (old version; from file )
        if self.openstreetmap and data.get("drone_position", None) is not None:
            self.target_hist.append(
                self.openstreetmap.scale_to_img(
                    data["drone_position"],
                    (self.openstreetmap.width_meters, self.openstreetmap.height_meters),
                )
            )

        # target_heading = None
        # target_relative_heading = None

        # Calculate extra data if target is known
        if (
            data.get("position", None) is not None
            and data.get("drone_position", None) is not None
            and data.get("heading", None) is not None
        ):
            target_heading = get_heading(data["position"], data["drone_position"])
            target_relative_heading = target_heading - data["heading"]
            target_distance = get_distance(data["position"], data["drone_position"])
            print(f"Sensor position & heading = {data['position']},{data['heading']}")
            print(
                f"Target distance & heading = {target_distance},{target_relative_heading}"
            )
            print(f"{self.expected_target_rssi=}")
            self.expected_target_rssi = env.sensor.observation(
                [target_distance, target_relative_heading, None, None], 0
            )[0]

        ax.clear()
        if self.openstreetmap is not None:
            self.openstreetmap.plot_map(axis1=ax)

        if separable:
            ax.set_title("Time = {}".format(time_step))
        else:
            abs_particles = np.moveaxis(abs_particles, 1, 0)
            ax.set_title(
                f"Time = {str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}"
            )

        # Plot particles
        if env.simulated or (self.openstreetmap and not self.target_only_map):
            for t in range(env.state.n_targets):
                particles_x, particles_y = pol2cart(
                    abs_particles[t, :, 0], np.radians(abs_particles[t, :, 1])
                )
                if self.transform is not None:
                    particles_x += self.transform[0]
                    particles_y += self.transform[1]
                if separable:
                    particle_color = separable_color_array[t][0]
                else:
                    particle_color = "salmon"
                (line1,) = ax.plot(
                    particles_x,
                    particles_y,
                    "o",
                    color=particle_color,
                    markersize=4,
                    markeredgecolor="black",
                    label="Particles",
                    alpha=0.4,
                    zorder=1,
                )

                # PLOT HEATMAP OVER STREET MAP
                if self.enable_heatmap and self.openstreetmap:
                    heatmap, xedges, yedges = np.histogram2d(
                        particles_x,
                        particles_y,
                        bins=(self.openstreetmap.xedges, self.openstreetmap.yedges),
                    )
                    heatmap = gaussian_filter(heatmap, sigma=8)
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    im = ax.imshow(
                        heatmap.T,
                        extent=extent,
                        origin="lower",
                        cmap="jet",
                        interpolation="nearest",
                        alpha=0.2,
                    )
                    # plt.colorbar(im)

                # PLOT CENTROIDS
                centroid_x = np.mean(particles_x)
                centroid_y = np.mean(particles_y)
                if separable:
                    centroid_color = separable_color_array[t][1]
                else:
                    centroid_color = "magenta"

                (line2,) = ax.plot(
                    centroid_x,
                    centroid_y,
                    "^",
                    color=centroid_color,
                    markeredgecolor="black",
                    label="Mean Estimate",
                    markersize=20,
                    zorder=7,
                )

                if t == 0:
                    lines.extend([line1, line2])
                else:
                    lines.extend([])

                target_class_name = f"Target {t} particles"
                for class_name, class_idx in self.class_map.items():
                    if t == class_idx:
                        target_class_name = f"{class_name} particles"

                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="white",
                        markersize=8,
                        markeredgecolor="black",
                        markerfacecolor=centroid_color,
                        label=target_class_name,
                    )
                )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="white",
                    markeredgecolor="black",
                    markerfacecolor="white",
                    label="Avg Estimates",
                    markersize=12,
                )
            )

        # Plot sensor
        if env.simulated or self.sensor_gps_hist:
            arrow_x = None
            arrow_y = None
            if self.sensor_gps_hist:
                temp_np = np.array(self.sensor_gps_hist)
                sensor_x = temp_np[:, 0]
                sensor_y = temp_np[:, 1]
                if data.get("heading", None) or data.get("previous_heading", None):
                    arrow_x, arrow_y = pol2cart(
                        6, np.radians(data.get("heading", data["previous_heading"]))
                    )
            else:
                sensor_x, sensor_y = pol2cart(
                    np.array(self.sensor_hist)[:, 0],
                    np.radians(np.array(self.sensor_hist)[:, 1]),
                )
                if self.transform is not None:
                    sensor_x += self.transform[0]
                    sensor_y += self.transform[1]

                arrow_x, arrow_y = pol2cart(6, np.radians(env.state.sensor_state[2]))
            if arrow_x and arrow_y:
                line4 = mpatches.FancyArrow(
                    sensor_x[-1],
                    sensor_y[-1],
                    arrow_x,
                    arrow_y,
                    width=5,
                    head_width=15,
                    head_length=15,
                    facecolor=sensor_color,
                    zorder=7,
                    # edgecolor="black",
                    label="Sensor",
                    linewidth=1,
                )
                ax.add_patch(line4)
                lines.extend([line4])
            if len(self.sensor_hist) > 1:
                ax.plot(
                    sensor_x[len(sensor_x) - self.history_length :],
                    sensor_y[len(sensor_x) - self.history_length :],
                    linewidth=5,
                    color=sensor_color,
                    # markeredgecolor="black",
                    # markersize=4,
                    zorder=6,
                    # path_effects=[pe.Stroke(linewidth=7, foreground='black')]
                )
            legend_elements.append(
                mpatches.Patch(
                    facecolor=sensor_color, edgecolor="black", label="Sensor"
                )
            )

        # Plot targets
        if self.target_hist or self.target_gps_hist:
            color_map=get_cmap("tab10").colors
            if env.simulated:
                n_target_hist = env.state.n_targets
            else:
                n_target_hist = len(self.target_gps_hist)
            for t in range(n_target_hist):
                if env.simulated:
                    target_x, target_y = pol2cart(
                        np.array(self.target_hist)[:, t, 0],
                        np.radians(np.array(self.target_hist)[:, t, 1]),
                    )
                else:
                    temp_np = np.array(list(self.target_gps_hist.values())[t])
                    # temp_np = np.array(self.target_hist)
                    target_x = temp_np[:, 0]
                    target_y = temp_np[:, 1]

                # if self.transform is not None:
                #     target_x += self.transform[0]
                #     target_y += self.transform[1]

                if len(target_x) > 1:
                    ax.plot(
                        target_x,
                        target_y,
                        linewidth=3.0,
                        color=color_map[t],
                        zorder=3,
                        markersize=4,
                    )

                (line5,) = ax.plot(
                    target_x[-1],
                    target_y[-1],
                    "X",
                    color=color_map[t],
                    markeredgecolor=color_map[t],
                    # label="Targets",
                    markersize=8,
                    zorder=3,
                )

                target_class_name = f"Target {t}"
                if self.target_gps_hist:
                    target_class_name = list(self.target_gps_hist)[t]
                ax.text(
                    target_x[-1],
                    target_y[-1],
                    f"{target_class_name}",
                    color=color_map[t],
                    fontsize=16,
                    fontweight="bold",
                )
                # lines.extend([line5])
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="X",
                        color="white",
                        markerfacecolor=color_map[t],
                        markeredgecolor=color_map[t],
                        label=target_class_name,
                        markersize=10,
                    )
                )

        # Legend
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=len(legend_elements),
        )

        # X/Y Limits
        if self.openstreetmap is None:
            map_width = map_distance * 2.1
            min_map = -1 * int(map_width / 2)
            max_map = int(map_width / 2)
            ax.set_xlim(min_map, max_map)
            ax.set_ylim(min_map, max_map)

        # Sidebar Text
        if sidebar:
            self.pf_stats["map_state"][-1][0]
            map_r = (
                f"{self.pf_stats['map_state'][-1][0]:.0f} m"
                if self.pf_stats["map_state"][-1][0] is not None
                else f"unknown"
            )

            if self.pf_stats["map_state"][-1][0] is not None:
                map_theta = self.pf_stats["map_state"][-1][1]
                map_theta = np.degrees(np.arctan2(np.sin(map_theta), np.cos(map_theta)))
                map_theta = f"{map_theta:.0f} deg"
            else:
                map_theta = f"unknown"

            # map_theta = np.degrees(np.arctan2(np.sin(map_theta),np.cos(map_theta)))
            rel_centroid_x, rel_centroid_y = env.get_particle_centroids()[t]
            centroid_r, centroid_theta = cart2pol(rel_centroid_x, rel_centroid_y)
            centroid_theta = np.degrees(centroid_theta)
            mean_estimate_str = (
                f"TARGET ESTIMATE (mean)\n"
                f"Distance = {centroid_r:.0f} m\n"
                f"Bearing = {centroid_theta:.0f} deg"
            )
            map_estimate_str = (
                f"TARGET ESTIMATE (MAP)\n"
                f"Distance = {map_r}\n"
                f"Bearing = {map_theta}"
            )

            # actual_str = r'$\bf{Actual}$''\n' # prettier format but adds ~0.04 seconds ???
            actual_heading_str = (
                f"{data.get('heading'):.0f} deg"
                if data.get("heading", None)
                else f"unknown"
            )
            actual_speed_str = (
                f"{data.get('action_taken')[1]:.2f} m/s"
                if data.get("action_taken", None)
                else f"unknown"
            )
            actual_str = (
                f"SENSOR ACTUAL\n"
                f"Heading = {actual_heading_str}\n"
                f"Speed = {actual_speed_str}"
            )

            proposal_heading_str = (
                f"{data.get('action_proposal')[0]:.0f} deg"
                if None not in data.get("action_proposal", (None, None))
                else f"unknown"
            )
            proposal_speed_str = (
                f"{data.get('action_proposal')[1]:.2f} m/s"
                if None not in data.get("action_proposal", (None, None))
                else f"unknown"
            )
            proposal_str = (
                f"SENSOR PROPOSAL\n"
                f"Heading = {proposal_heading_str}\n"
                f"Speed = {proposal_speed_str}"
            )

            rssi_observed_str = (
                f"{env.last_observation} dB" if env.last_observation else f"unknown"
            )
            rssi_expected_str = (
                f"{self.expected_target_rssi:.0f} dB"
                if self.expected_target_rssi
                else f"unknown"
            )
            rssi_difference_str = (
                f"Difference = {env.last_observation - self.expected_target_rssi:.0f} dB\n"
                if (env.last_observation and self.expected_target_rssi)
                else ""
            )
            rssi_mean_str = (
                f"{self.pf_stats['mean_hypothesis'][-1][0]:.0f} dB"
                if self.pf_stats["mean_hypothesis"][-1][0]
                else f"unknown"
            )
            rssi_map_str = (
                f"{self.pf_stats['map_hypothesis'][-1][0]:.0f} dB"
                if self.pf_stats["map_hypothesis"][-1][0]
                else f"unknown"
            )
            rssi_str = (
                f"RSSI\n"
                f"Observed = {rssi_observed_str}\n"
                f"Expected = {rssi_expected_str}\n"
                # f"{rssi_difference_str}"
                f"Mean estimate = {rssi_mean_str}\n"
                f"MAP estimate = {rssi_map_str}"
            )
            # rssi_str += 'Target heading = {} \n'.format(target_heading) if target_heading else ''
            # rssi_str += 'Target relative heading = {} \n'.format(target_relative_heading) if target_relative_heading else ''

            sidebar_str = (
                f"{mean_estimate_str}"
                f"\n\n===============\n\n"
                f"{map_estimate_str}"
                f"\n\n===============\n\n"
                f"{actual_str}"
                f"\n\n===============\n\n"
                f"{proposal_str}"
                f"\n\n===============\n\n"
                f"{rssi_str}"
            )

            if len(fig.texts) == 0:
                props = dict(boxstyle="round", facecolor="paleturquoise", alpha=0.5)
                text = fig.text(
                    1.04,
                    1,
                    sidebar_str,
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=props,
                )
            else:
                fig.texts[0].set_text(sidebar_str)

            # if len(fig.texts) == 0:
            #     props = dict(boxstyle="round", facecolor="mistyrose", alpha=0.5)
            #     text = fig.text(
            #         1.04,
            #         1,
            #         estimate_str,
            #         transform=ax.transAxes,
            #         fontsize=14,
            #         verticalalignment="top",
            #         bbox=props,
            #     )
            #     props = dict(boxstyle="round", facecolor="palegreen", alpha=0.5)
            #     text = fig.text(
            #         1.04,
            #         0.85,
            #         actual_str,
            #         transform=ax.transAxes,
            #         fontsize=14,
            #         verticalalignment="top",
            #         bbox=props,
            #     )
            #     props = dict(boxstyle="round", facecolor="paleturquoise", alpha=0.5)
            #     text = fig.text(
            #         1.04,
            #         0.70,
            #         proposal_str,
            #         transform=ax.transAxes,
            #         fontsize=14,
            #         verticalalignment="top",
            #         bbox=props,
            #     )
            #     props = dict(boxstyle="round", facecolor="khaki", alpha=0.5)
            #     text = fig.text(
            #         1.04,
            #         0.55,
            #         rssi_str,
            #         transform=ax.transAxes,
            #         fontsize=14,
            #         verticalalignment="top",
            #         bbox=props,
            #     )
            # else:
            #     fig.texts[0].set_text(estimate_str)
            #     fig.texts[1].set_text(actual_str)
            #     fig.texts[2].set_text(proposal_str)
            #     fig.texts[3].set_text(rssi_str)

        self.native_plot = "true" if time_step % self.plot_every_n == 0 else "false"
        if self.native_plot == "true":
            plt.draw()
            # plt.pause(0.001)
            fig.canvas.start_event_loop(0.001)
            fig.canvas.draw_idle()
        if self.make_gif == "true":
            png_filename = os.path.join(self.plot_dir, "png", f"{time_step}.png")
            print(f"saving plots in {png_filename}")
            plt.savefig(png_filename, bbox_inches="tight")

    def build_multitarget_plots(
        self,
        env,
        time_step=None,
        fig=None,
        axs=None,
        centroid_distance_error=None,
        selected_plots=[1, 2, 3, 4, 5],
        simulated=True,
        textstr=None,
    ):
        xp = env.state.target_state
        belief = env.pf.particles.reshape(len(env.pf.particles), env.state.n_targets, 4)
        abs_sensor = env.state.sensor_state

        abs_particles = env.get_absolute_particles()

        if simulated:
            abs_target = np.array(env.get_absolute_target())
        else:
            abs_target = None

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        self.history_length = 50

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

        plt.tight_layout()
        # Put space between plots
        plt.subplots_adjust(wspace=0.7, hspace=0.2)

        color_array = [
            ["salmon", "darkred", "red"],
            ["lightskyblue", "darkblue", "blue"],
        ]

        plot_count = 0
        if axs is None:
            axs = {}

        map_width = 600
        min_map = -1 * int(map_width / 2)
        max_map = int(map_width / 2)
        cell_size = int((max_map - min_map) / max_map)
        cell_size = 2
        xedges = np.arange(min_map, max_map + cell_size, cell_size)
        yedges = np.arange(min_map, max_map + cell_size, cell_size)

        if 1 in selected_plots:
            # Plot 1: Particle Plot (Polar)
            plot_count += 1
            if 1 not in axs:
                axs[1] = fig.add_subplot(1, len(selected_plots), plot_count, polar=True)
            ax = axs[1]
            ax.clear()

            for t in range(env.state.n_targets):
                # plot particles
                plot_theta = np.radians(belief[:, t, 1])
                plot_r = belief[:, t, 0]  # [row[0] for row in belief]

                ax.plot(
                    plot_theta,
                    plot_r,
                    "o",
                    color=color_array[t][0],
                    markersize=4,
                    markeredgecolor="black",
                    label="particles",
                    alpha=0.3,
                    zorder=1,
                )

                # plot targets
                plot_x_theta = np.radians(xp[t, 1])
                plot_x_r = xp[t, 0]

            ax.set_ylim(0, 300)
        if 2 in selected_plots:
            # Plot 2: Particle Plot (Polar) with Interpolation
            plot_count += 1
            if 2 not in axs:
                axs[2] = fig.add_subplot(1, len(selected_plots), plot_count, polar=True)
            ax = axs[2]

            for t in range(env.state.n_targets):
                # Create grid values first via histogram.
                nbins = 10
                plot_theta = np.radians(belief[:, t, 1])
                plot_r = belief[:, t, 0]  # [row[0] for row in belief]
                counts, xbins, ybins = np.histogram2d(plot_theta, plot_r, bins=nbins)
                # Make a meshgrid for theta, r values
                tm, rm = np.meshgrid(xbins[:-1], ybins[:-1])
                # Build contour plot
                ax.contourf(tm, rm, counts)
                # True position
                plot_x_theta = np.radians(xp[t, 1])
                plot_x_r = xp[t, 0]
                ax.plot(plot_x_theta, plot_x_r, "X")

            ax.set_ylim(0, 300)
        if 3 in selected_plots:
            # Plot 3: Heatmap Plot (Cartesian)
            plot_count += 1
            if 3 not in axs:
                axs[3] = fig.add_subplot(1, len(selected_plots), plot_count)
            ax = axs[3]

            # COMBINED; UNCOMMENT AFTER PAPER PLOT
            all_particles_x, all_particles_y = [], []

            for t in range(env.state.n_targets):
                cart = np.array(
                    list(map(pol2cart, belief[:, t, 0], np.radians(belief[:, t, 1])))
                )
                x = cart[:, 0]
                y = cart[:, 1]
                all_particles_x.extend(x)
                all_particles_y.extend(y)

            heatmap, xedges, yedges = np.histogram2d(
                all_particles_x, all_particles_y, bins=(xedges, yedges)
            )
            heatmap = gaussian_filter(heatmap, sigma=8)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(
                heatmap.T,
                extent=extent,
                origin="lower",
                cmap="jet",
                interpolation="nearest",
            )
            plt.colorbar(im)
            ax.set_xlim(min_map, max_map)
            ax.set_ylim(min_map, max_map)
        if 4 in selected_plots:
            # Plot 4: Absolute Polar coordinates
            plot_count += 1
            if 4 not in axs:
                axs[4] = fig.add_subplot(1, len(selected_plots), plot_count, polar=True)
            ax = axs[4]
            ax.clear()

            lines = (
                []
            )  # https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.legend.html
            for t in range(env.state.n_targets):
                particles_x, particles_y = pol2cart(
                    abs_particles[:, t, 0], np.radians(abs_particles[:, t, 1])
                )
                centroid_x = np.mean(particles_x)
                centroid_y = np.mean(particles_y)
                centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)
                target_r, target_theta, target_x, target_y = [], [], [], []

                for i in range(5):
                    target_r.append(
                        self.abs_target_hist[10 * (i + 1) - 1][
                            env.state.n_targets - 1 - t
                        ][0]
                    )
                    target_theta.append(
                        np.radians(
                            self.abs_target_hist[10 * (i + 1) - 1][
                                env.state.n_targets - 1 - t
                            ][1]
                        )
                    )
                target_x, target_y = pol2cart(target_r, target_theta)
                if len(self.target_hist) > 1:
                    ax.plot(
                        np.radians(np.array(self.target_hist)[:-1, t, 1]),
                        np.array(self.target_hist)[:-1, t, 0],
                        linewidth=4.0,
                        color="limegreen",
                        zorder=3,
                        markersize=12,
                    )

                (line0,) = ax.plot(
                    target_theta[4],
                    target_r[4],
                    "X",
                    color="limegreen",
                    markeredgecolor="black",
                    label="targets",
                    markersize=20,
                    zorder=4,
                )

                (line1,) = ax.plot(
                    np.radians(abs_particles[:, t, 1]),
                    abs_particles[:, t, 0],
                    "o",
                    color=color_array[t][0],
                    markersize=4,
                    markeredgecolor="black",
                    label="particles",
                    alpha=0.3,
                    zorder=1,
                )
                if t == 0:
                    lines.extend([line0, line1])
                else:
                    lines.extend([line0])

            if len(self.sensor_hist) > 1:
                ax.plot(
                    np.radians(np.array(self.sensor_hist)[:-1, 1]),
                    np.array(self.sensor_hist)[:-1, 0],
                    linewidth=4.0,
                    color="mediumorchid",
                    zorder=3,
                    markersize=12,
                )

            (line4,) = ax.plot(
                np.radians(self.sensor_hist[-1][1]),
                self.sensor_hist[-1][0],
                "H",
                color="mediumorchid",
                markeredgecolor="black",
                label="sensor",
                markersize=20,
                zorder=3,
            )
            lines.extend([line4])
            ax.legend(
                handles=lines,
                loc="center left",
                bbox_to_anchor=(1.08, 0.5),
                fancybox=True,
                shadow=True,
            )
            ax.set_ylim(0, 250)
        if 5 in selected_plots:
            # Plot 5: Absolute Cartesian coordinates
            plot_count += 1
            if 5 not in axs:
                axs[5] = fig.add_subplot(1, len(selected_plots), plot_count)
            ax = axs[5]

            xedges = np.arange(min_map, max_map, cell_size)
            yedges = np.arange(min_map, max_map, cell_size)
            heatmap_combined = None
            all_particles_x, all_particles_y = [], []
            for t in range(env.state.n_targets):
                particles_x, particles_y = pol2cart(
                    abs_particles[:, t, 0], np.radians(abs_particles[:, t, 1])
                )
                all_particles_x.extend(particles_x)
                all_particles_y.extend(particles_y)
                centroid_x = np.mean(particles_x)
                centroid_y = np.mean(particles_y)
                centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)
                target_r, target_theta, target_x, target_y = [], [], [], []
                for i in range(5):
                    target_r.append(self.abs_target_hist[10 * (i + 1) - 1][t][0])
                    target_theta.append(
                        np.radians(self.abs_target_hist[10 * (i + 1) - 1][t][1])
                    )
                target_x, target_y = pol2cart(target_r, target_theta)

                ax.plot(centroid_x, centroid_y, "*", label="centroid", markersize=12)

                ax.plot(target_x[4], target_y[4], "X", label="target", markersize=12)
            sensor_r, sensor_theta, sensor_x, sensor_y = [], [], [], []
            for i in range(5):
                sensor_r.append(self.abs_sensor_hist[10 * (i + 1) - 1][0])
                sensor_theta.append(
                    np.radians(self.abs_sensor_hist[10 * (i + 1) - 1][1])
                )
            sensor_x, sensor_y = pol2cart(sensor_r, sensor_theta)
            ax.plot(sensor_x[4], sensor_y[4], "p", label="sensor", markersize=12)

            heatmap, xedges, yedges = np.histogram2d(
                all_particles_x, all_particles_y, bins=(xedges, yedges)
            )
            heatmap = gaussian_filter(heatmap, sigma=8)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(
                heatmap.T,
                extent=extent,
                origin="lower",
                cmap="jet",
                interpolation="nearest",
            )
            plt.colorbar(im)

            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.2, 0.5),
                fancybox=True,
                shadow=True,
            )
            ax.set_xlim(min_map, max_map)
            ax.set_ylim(min_map, max_map)
        if 6 in selected_plots:
            # Plot 1: Particle Plot (Polar)
            plot_count += 1
            if 6 not in axs:
                axs[6] = fig.add_subplot(1, len(selected_plots), plot_count)
            ax = axs[6]
            ax.clear()

            for t in range(env.state.n_targets):
                # plot particles
                plot_theta = np.radians(belief[:, t, 1])
                plot_r = belief[:, t, 0]
                particles_x, particles_y = pol2cart(
                    belief[:, t, 0], np.radians(belief[:, t, 1])
                )
                ax.plot(
                    particles_x,
                    particles_y,
                    "o",
                    color=color_array[t][0],
                    markersize=4,
                    markeredgecolor="black",
                    label="particles",
                    alpha=0.3,
                    zorder=1,
                )

                # plot targets
                plot_x_theta = np.radians(xp[t, 1])
                plot_x_r = xp[t, 0]

            ax.set_xlim(min_map, max_map)
            ax.set_ylim(min_map, max_map)

            sensor_x, sensor_y = pol2cart(
                self.sensor_hist[-1][0], np.radians(self.sensor_hist[-1][1])
            )
        if 7 in selected_plots:
            plot_count += 1
            if 7 not in axs:
                axs[7] = fig.add_subplot(1, len(selected_plots), plot_count)
            ax = axs[7]
            ax.clear()

            lines = (
                []
            )  # https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.legend.html
            for t in range(env.state.n_targets):
                particles_x, particles_y = pol2cart(
                    abs_particles[:, t, 0], np.radians(abs_particles[:, t, 1])
                )
                centroid_x = np.mean(particles_x)
                centroid_y = np.mean(particles_y)
                centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)
                target_r, target_theta, target_x, target_y = [], [], [], []

                for i in range(5):
                    target_r.append(
                        self.abs_target_hist[10 * (i + 1) - 1][
                            env.state.n_targets - 1 - t
                        ][0]
                    )
                    target_theta.append(
                        np.radians(
                            self.abs_target_hist[10 * (i + 1) - 1][
                                env.state.n_targets - 1 - t
                            ][1]
                        )
                    )
                target_x, target_y = pol2cart(target_r, target_theta)
                target_x, target_y = pol2cart(
                    np.array(self.target_hist)[:, t, 0],
                    np.radians(np.array(self.target_hist)[:, t, 1]),
                )

                if len(self.target_hist) > 1:
                    ax.plot(
                        target_x[:-1],
                        target_y[:-1],
                        linewidth=4.0,
                        color="limegreen",
                        zorder=3,
                        markersize=12,
                    )

                (line0,) = ax.plot(
                    target_x[-1],
                    target_y[-1],
                    "X",
                    color="limegreen",
                    markeredgecolor="black",
                    label="targets",
                    markersize=20,
                    zorder=4,
                )

                (line1,) = ax.plot(
                    particles_x,
                    particles_y,
                    "o",
                    color=color_array[t][0],
                    markersize=4,
                    markeredgecolor="black",
                    label="particles",
                    alpha=0.3,
                    zorder=1,
                )
                # ax.plot(centroid_theta, centroid_r, '*', color=color_array[t][1],markeredgecolor='white', label='centroid', markersize=12, zorder=2)
                if t == 0:
                    lines.extend([line0, line1])
                else:
                    lines.extend([line0])

            sensor_x, sensor_y = pol2cart(
                np.array(self.sensor_hist)[:, 0],
                np.radians(np.array(self.sensor_hist)[:, 1]),
            )
            if len(self.sensor_hist) > 1:
                ax.plot(
                    sensor_x[:-1],
                    sensor_y[:-1],
                    linewidth=4.0,
                    color="mediumorchid",
                    zorder=3,
                    markersize=12,
                )

            (line4,) = ax.plot(
                sensor_x[-1],
                sensor_y[-1],
                "H",
                color="mediumorchid",
                markeredgecolor="black",
                label="sensor",
                markersize=20,
                zorder=3,
            )
            lines.extend([line4])
            ax.legend(
                handles=lines,
                loc="center left",
                bbox_to_anchor=(1.08, 0.5),
                fancybox=True,
                shadow=True,
            )

            ax.set_xlim(min_map, max_map)
            ax.set_ylim(min_map, max_map)
        if 8 in selected_plots:
            plot_count += 1
            if 8 not in axs:
                axs[8] = fig.add_subplot(1, len(selected_plots), plot_count)
            ax = axs[8]
            ax.clear()

            lines = (
                []
            )  # https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.legend.html
            for t in range(env.state.n_targets):
                particles_x, particles_y = pol2cart(
                    abs_particles[:, t, 0], np.radians(abs_particles[:, t, 1])
                )
                centroid_x = np.mean(particles_x)
                centroid_y = np.mean(particles_y)
                centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)

                (line1,) = ax.plot(
                    particles_x,
                    particles_y,
                    "o",
                    color=color_array[t][0],
                    markersize=4,
                    markeredgecolor="black",
                    label="particles",
                    alpha=0.3,
                    zorder=1,
                )
                if t == 0:
                    lines.extend([line1])
                else:
                    lines.extend([])

            sensor_x, sensor_y = pol2cart(
                np.array(self.sensor_hist)[:, 0],
                np.radians(np.array(self.sensor_hist)[:, 1]),
            )
            if len(self.sensor_hist) > 1:
                ax.plot(
                    sensor_x[:-1],
                    sensor_y[:-1],
                    linewidth=4.0,
                    color="mediumorchid",
                    zorder=3,
                    markersize=12,
                )

            (line4,) = ax.plot(
                sensor_x[-1],
                sensor_y[-1],
                "H",
                color="mediumorchid",
                markeredgecolor="black",
                label="sensor",
                markersize=20,
                zorder=3,
            )
            lines.extend([line4])
            ax.legend(
                handles=lines,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True,
                shadow=True,
                ncol=2,
            )

            ax.set_xlim(min_map, max_map)
            ax.set_ylim(min_map, max_map)
            if textstr:
                props = dict(boxstyle="round", facecolor="palegreen", alpha=0.5)
                ax.text(
                    1.04,
                    0.75,
                    textstr[0],
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=props,
                )
                props = dict(boxstyle="round", facecolor="paleturquoise", alpha=0.5)
                ax.text(
                    1.04,
                    0.5,
                    textstr[1],
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=props,
                )

        png_filename = "{}/png/{}.png".format(self.plot_dir, time_step)
        return axs

    def build_plots(
        self,
        xp=[],
        belief=[],
        abs_sensor=None,
        abs_target=None,
        abs_particles=None,
        time_step=None,
        fig=None,
        ax=None,
    ):
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
        grid_r, grid_theta = [], []
        plot_r = [row[0] for row in belief]
        plot_theta = np.radians(np.array([row[1] for row in belief]))
        plot_x_theta = np.radians(xp[1])
        plot_x_r = xp[0]
        ax.plot(plot_theta, plot_r, "ro")
        ax.plot(plot_x_theta, plot_x_r, "bo")
        ax.set_ylim(-150, 150)
        ax.set_title("iteration {}".format(time_step), fontsize=16)

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
        ax.plot(plot_x_theta, plot_x_r, "bo")
        ax.set_ylim(-150, 150)
        ax.set_title("Interpolated Belief".format(time_step), fontsize=16)

        # Plot 3: Heatmap Plot (Cartesian)
        ax = fig.add_subplot(1, 5, 3)
        cart = np.array(list(map(pol2cart, belief[:, 0], np.radians(belief[:, 1]))))
        x = cart[:, 0]
        y = cart[:, 1]
        xedges = np.arange(-150, 153, 3)
        yedges = np.arange(-150, 153, 3)
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        heatmap = gaussian_filter(heatmap, sigma=5)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin="lower", cmap="coolwarm")
        plt.colorbar(im)
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_title("Particle heatmap (relative to sensor)")

        # Plots 4 & 5: Absolute Particle/Sensor/Target Plot
        # particles/centroid coordinates
        particles_x, particles_y = pol2cart(
            abs_particles[:, 0], np.radians(abs_particles[:, 1])
        )
        centroid_x = np.mean(particles_x)
        centroid_y = np.mean(particles_y)
        centroid_r, centroid_theta = cart2pol(centroid_x, centroid_y)
        sensor_r, sensor_theta, sensor_x, sensor_y = [], [], [], []
        target_r, target_theta, target_x, target_y = [], [], [], []
        for i in range(5):
            sensor_r.append(self.abs_sensor_hist[10 * (i + 1) - 1][0])
            sensor_theta.append(np.radians(self.abs_sensor_hist[10 * (i + 1) - 1][1]))
            target_r.append(self.abs_target_hist[10 * (i + 1) - 1][0])
            target_theta.append(np.radians(self.abs_target_hist[10 * (i + 1) - 1][1]))
            x_val, y_val = pol2cart(sensor_r, sensor_theta)
            sensor_x.append(x_val)
            sensor_y.append(y_val)
            x_val, y_val = pol2cart(target_r, target_theta)
            target_x.append(x_val)
            target_y.append(y_val)

        # Plot 4: Absolute Polar coordinates
        ax = fig.add_subplot(1, 5, 4, polar=True)
        ax.plot(
            np.radians(abs_particles[:, 1]),
            abs_particles[:, 0],
            "ro",
            label="particles",
            alpha=0.5,
        )
        ax.plot(centroid_theta, centroid_r, "c*", label="centroid", markersize=12)
        ax.plot(sensor_theta[4], sensor_r[4], "gp", label="sensor", markersize=12)
        ax.plot(target_theta[4], target_r[4], "bX", label="target", markersize=12)
        for i in range(4):
            ax.plot(sensor_theta[i], sensor_r[i], "gp", markersize=6, alpha=0.75)
            ax.plot(target_theta[i], target_r[i], "bX", markersize=6, alpha=0.75)
        ax.legend()
        ax.set_title("Absolute positions (polar)".format(time_step), fontsize=16)

        # Plot 5: Absolute Cartesian coordinates
        ax = fig.add_subplot(1, 5, 5)
        xedges = np.arange(-100, 103, 3)
        yedges = np.arange(-100, 103, 3)
        heatmap, xedges, yedges = np.histogram2d(
            np.asarray(particles_x)[:, 0],
            np.asarray(particles_y)[:, 0],
            bins=(xedges, yedges),
        )
        heatmap = gaussian_filter(heatmap, sigma=2)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin="lower", cmap="coolwarm")
        plt.colorbar(im)
        ax.plot(centroid_x, centroid_y, "c*", label="centroid", markersize=12)
        ax.plot(sensor_x[4], sensor_y[4], "gp", label="sensor", markersize=12)
        ax.plot(target_x[4], target_y[4], "bX", label="target", markersize=12)
        for i in range(4):
            ax.plot(sensor_x[i], sensor_y[i], "gp", markersize=6, alpha=0.55)
            ax.plot(target_x[i], target_y[i], "bX", markersize=6, alpha=0.55)
        ax.legend()
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_title("Absolute positions (cartesian)".format(time_step), fontsize=16)

        (
            r_error,
            theta_error,
            heading_error,
            centroid_distance_error,
            rmse,
            mae,
        ) = tracking_error(abs_target, abs_particles)

        png_filename = os.path.join(self.plot_dir, "png", f"{time_step}.png")
        print(f"saving plots in {png_filename}")
        plt.savefig(png_filename)
        plt.close(fig)


def write_config_log(config, logdir):
    if isinstance(config, configparser.ConfigParser):
        config2log = {section: dict(config[section]) for section in config.sections()}
    else:
        config2log = dict(config)

    # write config to file
    config_filename = f"{logdir}config.log"
    with open(config_filename, "w", encoding="UTF-8") as f:
        f.write(json.dumps(config2log))


def read_header_log(filename):
    with open(filename, "r", encoding="UTF-8") as f:
        config = json.load(f)
    return config


def particles_mean_belief(particles):
    particles_r = particles[:, 0]
    particles_theta = np.radians(particles[:, 1])
    particles_x, particles_y = pol2cart(particles_r, particles_theta)

    # centroid of particles x,y
    mean_x = np.mean(particles_x)
    mean_y = np.mean(particles_y)

    # centroid of particles r,theta
    mean_r, mean_theta = cart2pol(mean_x, mean_y)

    particles_heading = particles[:, 2]
    particles_heading_rad = np.radians(particles_heading)
    mean_heading_rad = np.arctan2(
        np.mean(np.sin(particles_heading_rad)), np.mean(np.cos(particles_heading_rad))
    )
    mean_heading = np.degrees(mean_heading_rad)

    mean_spd = np.mean(particles[:, 3])

    return (
        particles_x,
        particles_y,
        mean_x,
        mean_y,
        mean_r,
        mean_theta,
        mean_heading,
        mean_spd,
    )


def particles_centroid_xy(particles):
    particles_r = particles[:, 0]
    particles_theta = np.radians(particles[:, 1])
    particles_x, particles_y = pol2cart(particles_r, particles_theta)

    # centroid of particles x,y
    mean_x = np.mean(particles_x)
    mean_y = np.mean(particles_y)

    return [mean_x, mean_y]


def angle_diff(angle):
    diff = angle % 360

    diff = (diff + 360) % 360

    diff[diff > 180] -= 360
    return diff


def tracking_error(all_targets, all_particles):
    """
    Calculate different tracking errors
    """
    results = []
    r_error = None
    theta_error = None
    heading_error = None
    centroid_distance_error = None
    rmse = None
    mae = None
    n_targets = len(all_particles[0]) // 4

    # reorder targets to fit closest particles
    min_distance = None
    optimal_target_permutation = None

    for idxs in list(permutations(range(n_targets))):
        target_permutation = all_targets[list(idxs)]

        distance = 0
        for t in range(n_targets):
            particle_centroid = np.array(
                particles_centroid_xy(all_particles[:, 4 * t : 4 * (t + 1)])
            )
            target = np.array(
                pol2cart(target_permutation[t][0], np.radians(target_permutation[t][1]))
            )
            distance += np.linalg.norm(particle_centroid - target) ** 2
        if min_distance is None or distance < min_distance:
            min_distance = distance
            optimal_target_permutation = target_permutation

    for t in range(n_targets):
        target = optimal_target_permutation[t]
        particles = all_particles[:, 4 * t : 4 * (t + 1)]

        target_r = target[0]
        target_theta = np.radians(target[1])
        target_heading = target[2]
        target_x, target_y = pol2cart(target_r, target_theta)

        (
            particles_x,
            particles_y,
            mean_x,
            mean_y,
            mean_r,
            mean_theta,
            mean_heading,
            mean_spd,
        ) = particles_mean_belief(particles)

        r_error = np.mean(np.abs(target_r - particles[:, 0]))
        theta_error = np.mean(np.abs(angle_diff(target[1] - particles[:, 1])))
        heading_diff = np.abs(np.mean(target_heading - particles[:, 2])) % 360
        heading_error = heading_diff if heading_diff <= 180 else 360 - heading_diff

        # centroid euclidean distance error x,y
        centroid_distance_error = np.sqrt(
            (mean_x - target_x) ** 2 + (mean_y - target_y) ** 2
        )

        mae = np.mean(
            np.sqrt((particles_x - target_x) ** 2 + (particles_y - target_y) ** 2)
        )

        # root mean square error
        rmse = np.sqrt(
            np.mean((particles_x - target_x) ** 2 + (particles_y - target_y) ** 2)
        )

        results.append(
            [r_error, theta_error, heading_error, centroid_distance_error, rmse, mae]
        )
    results = np.array(results).T

    if len(results) > 5:
        r_error = results[0]
        theta_error = results[1]
        heading_error = results[2]
        centroid_distance_error = results[3]
        rmse = results[4]
        mae = results[5]

    return r_error, theta_error, heading_error, centroid_distance_error, rmse, mae


def tracking_metrics_separable(all_targets, all_particles):
    """
    Calculate different tracking metrics
    """
    results = []
    r_error = None
    theta_error = None
    heading_error = None
    centroid_distance_error = None
    rmse = None
    mae = None
    n_targets, n_particles, n_states = all_particles.shape

    for t in range(n_targets):
        target = all_targets[t]
        particles = all_particles[t]

        target_r = target[0]
        target_theta = np.radians(target[1])
        target_heading = target[2]
        target_x, target_y = pol2cart(target_r, target_theta)

        (
            particles_x,
            particles_y,
            mean_x,
            mean_y,
            mean_r,
            mean_theta,
            mean_heading,
            mean_spd,
        ) = particles_mean_belief(particles)

        r_error = np.mean(np.abs(target_r - particles[:, 0]))
        theta_error = np.mean(np.abs(angle_diff(target[1] - particles[:, 1])))
        heading_diff = np.abs(np.mean(target_heading - particles[:, 2])) % 360
        heading_error = heading_diff if heading_diff <= 180 else 360 - heading_diff

        # centroid euclidean distance error x,y
        centroid_distance_error = np.sqrt(
            (mean_x - target_x) ** 2 + (mean_y - target_y) ** 2
        )

        mae = np.mean(
            np.sqrt((particles_x - target_x) ** 2 + (particles_y - target_y) ** 2)
        )

        # root mean square error
        rmse = np.sqrt(
            np.mean((particles_x - target_x) ** 2 + (particles_y - target_y) ** 2)
        )

        results.append(
            [r_error, theta_error, heading_error, centroid_distance_error, rmse, mae]
        )
    results = np.array(results).T

    r_error = results[0]
    theta_error = results[1]
    heading_error = results[2]
    centroid_distance_error = results[3]
    rmse = results[4]
    mae = results[5]

    return r_error, theta_error, heading_error, centroid_distance_error, rmse, mae
