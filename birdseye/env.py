import numpy as np
import random
from .utils import pol2cart, particles_mean_belief
from birdseye.pfrnn.pfrnn import pfrnn
from pfilter import ParticleFilter, systematic_resample


class RFEnv(object):

    def __init__(self, sensor=None, actions=None, state=None, simulated=False):
        # Sensor definitions
        self.sensor = sensor
        # Action space and function to convert from action to index and vice versa
        self.actions = actions
        # Setup initial state
        self.state = state
        # Flag for simulation vs real data
        self.simulated = simulated

        self.pfrnn = pfrnn()

    def dynamics(self, particles, control=None, **kwargs):
        """Helper function for particle filter dynamics

        Returns
        -------
        array_like
            Updated particle state information
        """
        return np.array([list(self.state.update_state(p, control)) for p in particles])

    def reset(self, num_particles=4000):
        """Reset initial state and particle filter

        Parameters
        ----------
        num_particles : integer
            Number of particles to build particle filter

        Returns
        -------
        env_obs : array_like
            Heatmap distribution of observed particles from reset filter
        """

        self.iters = 0
        self.state.target_state = self.state.init_target_state()
        self.state.sensor_state = self.state.init_sensor_state()

        # Setup particle filter
        self.pf = ParticleFilter(
                        prior_fn=lambda n: np.array([self.state.random_state() for i in range(n)]),
                        observe_fn=lambda states, **kwargs: np.array([np.array(self.sensor.observation(x)) for x in states]),
                        n_particles=num_particles,
                        dynamics_fn=self.dynamics,
                        noise_fn=lambda x, **kwargs: x,
                        resample_proportion=0.005,
                        #noise_fn=lambda x:
                        #            gaussian_noise(x, sigmas=[0.2, 0.2, 0.1, 0.05, 0.05]),
                        weight_fn=lambda hyp, o, xp=None,**kwargs: [self.sensor.weight(None, o, state=x) for x in xp],
                        resample_fn=systematic_resample,
                        column_names = ['range', 'bearing', 'relative_course', 'own_speed'])

        env_obs = self.env_observation()
        return env_obs


    # returns observation, reward, done, info
    def step(self, action_idx):
        """Function to make step based on
           state variables and action index

        Parameters
        ----------
        action_idx : integer
            Index for action to make step

        Returns
        -------
        env_obs : array_like
            Heatmap distribution of observed particles from filter
        reward : float
            Reward value for specified action
        0 : int
            Placeholder integer value
        info : dict
            Dictionary to track step specific values (reward, iteration)
        """

        # Get action based on index
        action = self.actions.index_to_action(action_idx)
        # Determine next state based on action & current state variables
        next_state = self.state.update_state(self.state.target_state, action)
        # Update absolute position of sensor
        self.state.update_sensor(action)
        # Get sensor observation
        observation = self.sensor.observation(next_state)
        # Update particle filter
        self.pf.update(np.array(observation), xp=self.pf.particles, control=action)
        # Calculate reward based on updated state & action
        reward = self.state.reward_func(state=next_state, action_idx=action_idx, particles=self.pf.particles)
        #reward = -1. * self.get_distance_error()
        # Update the state variables
        self.state.target_state = next_state

        env_obs = self.env_observation()
        self.iters += 1
        info = {'episode':{}}
        info['episode']['l'] = self.iters
        info['episode']['r'] = reward
        info['observation'] = observation

        return (env_obs, reward, 0, info)

    def entropy_collision_reward(self, state, action_idx=None, delta=10, collision_weight=1):
        pf_r = self.pf.particles[:,0]
        pf_theta = np.radians(self.pf.particles[:,1])
        pf_x, pf_y = pol2cart(pf_r, pf_theta)
        xedges = np.arange(-150, 153, 3)
        yedges = np.arange(-150, 153, 3)
        b = np.histogram2d(pf_x, pf_y, bins=(xedges, yedges))
        b /= np.sum(b)
        b += 0.0000001

        H = -1. * np.sum([b * np.log(b)])
        collision_rate = np.mean(self.pf.particles[:,0] < delta)
        cost = H + collision_weight * collision_rate

        return -1. * cost

    def env_observation(self):
        """Helper function for environment observation

        Returns
        -------
        array_like
            Heatmap distribution of current observed particles
        """
        #return np.expand_dims(self.particle_heatmap_obs(self.pf.particles), axis=0)
        pf_map = np.expand_dims(self.particle_heatmap_obs(self.pf.particles), axis=0).reshape(-1)
        _,_,_,_, mean_r, mean_theta, mean_heading, mean_spd = particles_mean_belief(self.pf.particles)
        mean_belief = np.array([mean_r, mean_theta, mean_heading, mean_spd])
        return np.concatenate((mean_belief, pf_map))

    def particle_heatmap_obs(self, belief):
        """Function to build histogram representing
           belief distribution in cart coords

        Parameters
        ----------
        belief : array_like
            Belief distribution parameters

        Returns
        -------
        heatmap : array_like
            Histogram of belief state
        """
        # Transformation of belief to cartesian coords
        cart  = np.array(list(map(pol2cart, belief[:,0], np.radians(belief[:,1]))))
        x = cart[:,0]
        y = cart[:,1]

        # Build two-dim histogram distribution
        xedges = np.arange(-150, 153, 3)
        yedges = np.arange(-150, 153, 3)
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))

        return heatmap

    def get_absolute_particles(self):
        return np.array([self.state.get_absolute_state(t) for t in self.pf.particles])

    def get_absolute_target(self):
        return self.state.get_absolute_state(self.state.target_state)

    def get_particle_centroid(self):

        particles = self.pf.particles

        particles_r = particles[:,0]
        particles_theta = np.radians(particles[:,1])
        particles_heading = particles[:,2]
        particles_x, particles_y = pol2cart(particles_r, particles_theta)

        # centroid of particles x,y
        mean_x = np.mean(particles_x)
        mean_y = np.mean(particles_y)

        return mean_x, mean_y

    def get_distance_error(self):

        mean_x, mean_y = self.get_particle_centroid()

        target_r = self.state.target_state[0]
        target_theta = np.radians(self.state.target_state[1])
        target_x, target_y = pol2cart(target_r, target_theta)

        centroid_distance_error = np.sqrt((mean_x - target_x)**2 + (mean_y - target_y)**2)

        return centroid_distance_error
