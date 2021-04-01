import numpy as np 
import random 
from .utils import pol2cart
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

    def dynamics(self, particles, control=None, **kwargs):
        """Helper function for particle filter dynamics

        Returns
        -------
        array_like
            Updated particle state information
        """
        return np.array([list(self.state.update_state(p, control)) for p in particles])

    def reset(self, num_particles=500):
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
        self.state.state_vars = self.state.init_state()

        # Setup particle filter
        self.pf = ParticleFilter(
                        prior_fn=lambda n: np.array([self.sensor.near_state(self.state.state_vars) for i in range(n)]),
                        observe_fn=lambda states, **kwargs: np.array([np.array(self.sensor.observation(x)) for x in states]),
                        n_particles=num_particles,
                        dynamics_fn=self.dynamics,
                        noise_fn=lambda x, **kwargs: x,
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
        next_state = self.update_state(self.state.state_vars, action)
        # Get sensor observation
        observation = self.sensor.observation(next_state)
        # Update particle filter
        self.pf.update(np.array(observation), xp=self.pf.particles, control=action)
        # Calculate reward based on updated state & action
        reward = self.state.reward_func(next_state, action_idx)
        # Update the state variables
        self.state.state_vars = next_state

        env_obs = self.env_observation()
        self.iters += 1 
        info = {'episode':{}}
        info['episode']['l'] = self.iters
        info['episode']['r'] = reward

        return (env_obs, reward, 0, info)

    def env_observation(self): 
        """Helper function for environment observation

        Returns
        -------
        array_like
            Heatmap distribution of current observed particles
        """
        return np.expand_dims(self.particle_heatmap_obs(self.pf.particles), axis=0)
        
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
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)

        return heatmap
