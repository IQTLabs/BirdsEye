import numpy as np
from pfilter import ParticleFilter
from pfilter import systematic_resample
from scipy.ndimage.filters import gaussian_filter
from timeit import default_timer as timer

#from .pfrnn.pfrnn import pfrnn
from .utils import particle_swap
from .utils import particles_mean_belief
from .utils import pol2cart

class RFMultiSeparableEnv:
    def __init__(self, sensor=None, actions=None, state=None, simulated=True, num_particles=2000):
        # Sensor definitions
        self.sensor = sensor
        # Action space and function to convert from action to index and vice versa
        self.actions = actions
        # Setup initial state
        self.state = state
        # Flag for simulation vs real data
        self.simulated = simulated
        self.n_particles = num_particles 

        #self.pfrnn = pfrnn()

        self.last_observation = None
        self.pf = None
        self.iters = 0

    def dynamics(
        self,
        particles,
        control=None,
        distance=None,
        course=None,
        heading=None,
        **kwargs
    ):
        """Helper function for particle filter dynamics

        Returns
        -------
        array_like
            Updated particle state information
        """
        start = timer() 
        n_particles, n_states = particles.shape
        assert n_states == self.state.state_dim

        updated_particles = []
        if not self.simulated:
            for p in range(n_particles):
                updated_particles.append(
                    self.state.update_state(
                        particles[p],
                        control=control,
                        distance=distance,
                        course=course,
                        heading=heading,
                    )
                )
        else: 
            updated_particles = self.state.update_state_vectorized(particles, control=control)
        # if not np.allclose(updated_particles, updated_particles2): 
        # #if not np.all(updated_particles==updated_particles2): 
        #     print(f"{updated_particles=}")
        #     print(f"{updated_particles2=}")
        #     print(updated_particles==updated_particles2)
        end = timer() 
        #print(f"dynamics: {end-start}")
        return np.array(updated_particles)

        

    def particle_noise(self, particles, sigmas=[1, 2, 2], xp=None):
        start = timer() 
        n_particles, n_states = particles.shape
        assert n_states == self.state.state_dim

        # particles[:,0] += np.random.normal(0, sigmas[0], (n_particles))
        # particles[:,0] = np.clip(particles[:,0], a_min=1, a_max=None)
        # particles[:,1] += np.random.normal(0, sigmas[1], (n_particles))
        # particles[:,2] += np.random.normal(0, sigmas[2], (n_particles))
        particles[:,[0,1,2]] += np.random.normal([0,0,0],sigmas, (n_particles,3))
        particles[:,0] = np.clip(particles[:,0], a_min=1, a_max=None)
        end = timer() 
        #print(f"noise = {end-start}")
        return particles

    def reset(self,):
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
        if self.simulated:
            self.state.target_state = self.state.init_target_state()
        self.state.sensor_state = self.state.init_sensor_state()

        self.pf = []
        for t in range(self.state.n_targets):
            target_pf = ParticleFilter(
                prior_fn=lambda n: np.array(
                    [
                        self.state.random_particle_state()
                        for _ in range(n)
                    ]
                ),
                # observe_fn=lambda states, **kwargs: np.array(
                #     [
                #         self.sensor.observation(
                #             x, 
                #             t,
                #             fading_sigma=0
                #         )
                #         for x in states
                #     ]
                # ),
                observe_fn=lambda states, **kwargs: np.array(
                    self.sensor.observation_vectorized(states, t)
                ),
                n_particles=self.n_particles,
                dynamics_fn=self.dynamics,
                resample_proportion=0.005,
                noise_fn=lambda x, **kwargs: self.particle_noise(x, sigmas=[1, 2, 2]),
                weight_fn=lambda hyp, o, xp=None, **kwargs: self.sensor.weight(hyp, o),
                resample_fn=systematic_resample,
                n_eff_threshold=1,
                column_names=["range", "heading", "relative_course", "own_speed"],
            )
            self.pf.append(target_pf)

    def pffilter_copy(self, pf, n_downsample=None): 
        """Modified from https://github.com/johnhw/pfilter/blob/master/pfilter/pfilter.py, because missing noise_fn
        Copy this filter at its current state. Returns
        an exact copy, that can be run forward indepedently of the first.
        Beware that if your passed in functions (e.g. dynamics) are stateful, behaviour
        might not be independent! (tip: write stateless functions!)
        Returns:
        ---------
            A new, independent copy of this filter.
        """
        # construct the filter
        new_copy = ParticleFilter(
            observe_fn=pf.observe_fn,
            resample_fn=pf.resample_fn,
            n_particles=pf.n_particles,
            prior_fn=pf.prior_fn,
            dynamics_fn=pf.dynamics_fn,
            noise_fn=pf.noise_fn,
            weight_fn=pf.weight_fn,
            resample_proportion=pf.resample_proportion,
            column_names=pf.column_names,
            internal_weight_fn=pf.internal_weight_fn,
            transform_fn=pf.transform_fn,
            n_eff_threshold=pf.n_eff_threshold,
        )

        # copy particle state
        for array in ["particles", "original_particles", "original_weights", "weights"]:
            setattr(new_copy, array, np.array(getattr(pf, array)))

        # copy any attributes
        for array in [
            "mean_hypothesis",
            "mean_state",
            "map_state",
            "map_hypothesis",
            "hypotheses",
            "n_eff",
            "weight_informational_energy",
            "weight_entropy",
        ]:
            if hasattr(pf, array):
                setattr(new_copy, array, getattr(pf, array).copy())

        if n_downsample: 
            new_copy.n_particles = n_downsample
            new_copy.weights = np.ones(n_downsample) / n_downsample
            new_copy.particles = new_copy.particles[np.random.randint(len(new_copy.particles), size=n_downsample)]
        return new_copy

    def pf_copy(self, n_downsample=None): 
        return [self.pffilter_copy(pf, n_downsample=n_downsample) for pf in self.pf]
    
    def random_state(self, pf): 
        state = ([
            pf[i].particles[np.random.choice(pf[i].particles.shape[0], 1, False)][0] 
            for i in range(self.state.n_targets)
        ])
        return state 
    
    # returns observation, reward, done, info
    def real_step(self, data):
        # action = data['action_taken'] if data.get('action_taken', None) else (0,0)

        if not data["needs_processing"]: 
            data["distance"] = None
            data["course"] = None

        distance = data["distance"]
        course = data["course"]
        heading = data["heading"]
        data["needs_processing"] = False

        # Update position of sensor
        self.state.update_real_sensor(
            distance,
            course,
            heading,
            #data.get("distance", None),
            #data.get("course", None),
            #data.get("heading", None),
        )

        # Get sensor observation
        observation = self.sensor.real_observation()
        observation = np.array(observation) #if observation is not None else None

        # Update particle filter
        for t in range(self.state.n_targets):
            self.pf[t].update(
                observation[t],
                xp=self.pf[t].particles,
                distance=distance,
                course=course,
                heading=heading,
                # distance=data.get("distance", None),
                # course=data.get("course", None),
                # heading=data.get("heading", None),
            )
        # particle_swap(self)

        # Calculate reward based on updated state & action
        control_heading = (
            heading if heading is not None else self.state.sensor_state[2]
        )
        control_delta_heading = (control_heading - self.state.sensor_state[2]) % 360
        # reward = self.state.reward_func(
        #     state=None,
        #     action=(control_delta_heading, data.get("distance", 0)),
        #     particles=self.pf.particles,
        # )
        reward = None

        #belief_obs = self.env_observation()
        belief_obs = None

        self.last_observation = observation

        #return (belief_obs, reward, observation)
        return observation

    def void_probability(self, actions, r_min, min_bound=0.8):

        p_outside_void = []
        updated_particles = [self.pf[t].particles.copy() for t in range(self.state.n_targets)]
        for action in actions:
            for t in range(self.state.n_targets):
                target_particles = self.dynamics(updated_particles[t], control=action)
                updated_particles[t] = target_particles
                B = 1-np.mean(target_particles[:,0] < r_min)
                p_outside_void.append(B)
                #print(f"probability outside void = {B}")
        updated_particles = np.array(updated_particles)
        if np.min(p_outside_void) >= min_bound:
            return True, updated_particles
        return False, updated_particles

    # returns observation, reward, done, info
    def step(self, action):
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
        #action = self.actions.index_to_action(action_idx)
        # Determine next state based on action & current state variables
        # next_state = np.array(
        #     [
        #         self.state.update_state(target_state, action)
        #         for target_state in self.state.target_state
        #     ]
        # )
        next_state = self.state.update_state_vectorized(np.array(self.state.target_state), control=action)
        # Update absolute position of sensor
        self.state.update_sensor(action)

        observations = []
        for t in range(self.state.n_targets):
            # Get sensor observation
            observation = self.sensor.observation(next_state[t], t)
            observations.append(observation)
            # Update particle filter

            self.pf[t].update(np.array(observation), xp=self.pf[t].particles, control=action)
            #particle_swap(self)

        # Calculate reward based on updated state & action
        reward = None
        # reward = self.state.reward_func(
        #     state=next_state, action=action, particles=self.pf.particles
        # )
        # reward = -1. * self.get_distance_error()

        # Update the state variables
        self.state.target_state = next_state

        #env_obs = self.env_observation()
        env_obs = None
        self.iters += 1
        info = {"episode": {}}
        info["episode"]["l"] = self.iters
        info["episode"]["r"] = reward
        info["observation"] = observations

        return (env_obs, reward, 0, info)


    def env_observation(self):
        """Helper function for environment observation

        Returns
        -------
        array_like
            Heatmap distribution of current observed particles
        """
        # return np.expand_dims(self.particle_heatmap_obs(self.pf.particles), axis=0)
        belief = self.pf.particles.reshape(
            len(self.pf.particles), self.state.n_targets, 4
        )
        # flattened pf map [2 x 100 x 100] -> [20000]
        pf_map = self.particle_heatmap_obs(belief).reshape(-1)
        mean_belief = []
        for t in range(self.state.n_targets):
            (
                _,
                _,
                _,
                _,
                mean_r,
                mean_theta,
                mean_heading,
                mean_spd,
            ) = particles_mean_belief(belief[:, t, :])
            mean_belief.extend([mean_r, mean_theta, mean_heading, mean_spd])
        # flattened mean belief [2 x 4] -> [8]
        mean_belief = np.array(mean_belief)

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
        heatmaps = []
        map_width = 600
        min_map = -1 * int(map_width / 2)
        max_map = int(map_width / 2)
        cell_size = 2  # (max_map - min_map)/max_map
        xedges = np.arange(min_map, max_map + cell_size, cell_size)
        yedges = np.arange(min_map, max_map + cell_size, cell_size)
        for t in range(self.state.n_targets):
            cart = np.array(
                list(map(pol2cart, belief[:, t, 0], np.radians(belief[:, t, 1])))
            )
            x = cart[:, 0]
            y = cart[:, 1]

            # Build two-dim histogram distribution
            h, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
            h = gaussian_filter(h, sigma=8)
            heatmaps.append(h)
        heatmaps = np.array(heatmaps)
        return heatmaps

    def get_absolute_particles(self):
        return np.array(
            [
                [
                    self.state.get_absolute_state(p)
                    for p in self.pf[t].particles
                ]
                for t in range(self.state.n_targets)
            ]
        )

    def get_absolute_target(self):
        return np.array(
            [self.state.get_absolute_state(state) for state in self.state.target_state]
        )

    def get_particle_centroids(self, particles=None):
        centroids = []
        if particles is None:
            for t in range(self.state.n_targets):
                particles_x, particles_y = pol2cart(
                    self.pf[t].particles[:,0], 
                    np.radians(self.pf[t].particles[:,1])
                )
                centroids.append([np.mean(particles_x), np.mean(particles_y)])
        else:
            n_targets, n_particles, n_states = particles.shape

            assert n_targets == self.state.n_targets 

            for t in range(n_targets):
                particles_x, particles_y = pol2cart(
                    particles[t,:,0], 
                    np.radians(particles[t,:,1])
                )
                centroids.append([np.mean(particles_x), np.mean(particles_y)])

        return np.array(centroids)

    def get_particle_std_dev_cartesian(self, particles=None):
        std_dev = []
        if particles is None:
            for t in range(self.state.n_targets):
                particles_x, particles_y = pol2cart(
                    self.pf[t].particles[:,0], 
                    np.radians(self.pf[t].particles[:,1])
                )
                std_dev.append([np.std(particles_x), np.std(particles_y)])
        else:
            n_targets, n_particles, n_states = particles.shape
            assert n_targets == self.state.n_targets 

            for t in range(n_targets):
                particles_x, particles_y = pol2cart(
                    particles[t,:,0], 
                    np.radians(particles[t,:,1])
                )
                std_dev.append([np.std(particles_x), np.std(particles_y)])

        return np.array(std_dev)

    def get_particle_std_dev_polar(self, particles=None):
        std_dev = []
        if particles is None:
            for t in range(self.state.n_targets):
                std_dev.append([np.std(self.pf[t].particles[:,0]), np.std(self.pf[t].particles[:,1])])
        else:
            n_targets, n_particles, n_states = particles.shape
            assert n_targets == self.state.n_targets 
            
            for t in range(n_targets):
                std_dev.append([np.std(particles[t,:,0]), np.std(particles[t,:,1])])

        return np.array(std_dev)

    def get_all_particles(self):
        return np.array(
            [
                self.pf[t].particles 
                for t in range(self.state.n_targets)
            ]
        )

class RFMultiEnv:
    def __init__(self, sensor=None, actions=None, state=None, simulated=True):
        # Sensor definitions
        self.sensor = sensor
        # Action space and function to convert from action to index and vice versa
        self.actions = actions
        # Setup initial state
        self.state = state
        # Flag for simulation vs real data
        self.simulated = simulated

        self.pfrnn = pfrnn()

        self.last_observation = None
        self.pf = None
        self.iters = 0

    def dynamics(
        self,
        particles,
        control=None,
        distance=None,
        course=None,
        heading=None,
        **kwargs
    ):
        """Helper function for particle filter dynamics

        Returns
        -------
        array_like
            Updated particle state information
        """
        updated_particles = []
        for p in particles:
            new_p = []
            for t in range(self.state.n_targets):
                new_p += self.state.update_state(
                    p[4 * t : 4 * (t + 1)],
                    control=control,
                    distance=distance,
                    course=course,
                    heading=heading,
                )
            # new_p = np.array([self.state.update_state(target_state, control) for target_state in p])
            updated_particles.append(new_p)
        return np.array(updated_particles)

    def particle_noise(self, particles, sigmas=[1, 2, 2], xp=None):

        for t in range(self.state.n_targets):
            particles[:, [4 * t]] += np.random.normal(0, sigmas[0], (len(particles), 1))
            particles[:, [4 * t]] = np.clip(particles[:, [4 * t]], a_min=1, a_max=None)
            particles[:, [(4 * t) + 1]] += np.random.normal(
                0, sigmas[1], (len(particles), 1)
            )
            particles[:, [(4 * t) + 2]] += np.random.normal(
                0, sigmas[2], (len(particles), 1)
            )

            # particles[:,[0,4]] += np.random.normal(0,sigmas[0], (len(particles), 2))
            # particles[:,[0,4]] = np.clip(particles[:,[0,4]], a_min=1, a_max=None)
            # particles[:,[1,5]] += np.random.normal(0,sigmas[1], (len(particles), 2))
            # particles[:,[2,6]] += np.random.normal(0,sigmas[2], (len(particles), 2))

        return particles

    def reset(self, num_particles=2000):
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
        if self.simulated:
            self.state.target_state = self.state.init_target_state()
        self.state.sensor_state = self.state.init_sensor_state()

        # Setup particle filter
        self.pf = ParticleFilter(
            prior_fn=lambda n: np.array(
                [
                    np.array(self.state.init_particle_state()).reshape(-1)
                    for i in range(n)
                ]
            ),
            observe_fn=lambda states, **kwargs: np.array(
                [
                    self.sensor.observation(
                        [x[4 * t : 4 * (t + 1)] for t in range(self.state.n_targets)], 
                        fading_sigma=0
                    )
                    for x in states
                ]
            ),
            n_particles=num_particles,
            dynamics_fn=self.dynamics,
            resample_proportion=0.005,  # 0.005,
            # noise_fn=lambda x, **kwargs: x,
            noise_fn=lambda x, **kwargs: self.particle_noise(x, sigmas=[1, 2, 2]),
            #            gaussian_noise(x, sigmas=[0.2, 0.2, 0.1, 0.05, 0.05]),
            # [self.sensor.weight(None, o, state=x) for x in xp],
            weight_fn=lambda hyp, o, xp=None, **kwargs: self.sensor.weight(hyp, o),
            resample_fn=systematic_resample,
            n_eff_threshold=1,
            column_names=["range", "heading", "relative_course", "own_speed"],
        )

        env_obs = self.env_observation()
        return env_obs

    # returns observation, reward, done, info
    def real_step(self, data):
        # action = data['action_taken'] if data.get('action_taken', None) else (0,0)

        # Update position of sensor
        self.state.update_real_sensor(
            data.get("distance", None),
            data.get("course", None),
            data.get("heading", None),
        )

        # Get sensor observation
        observation = self.sensor.real_observation()
        observation = np.array(observation) if observation is not None else None

        # Update particle filter
        self.pf.update(
            observation,
            xp=self.pf.particles,
            distance=data.get("distance", None),
            course=data.get("course", None),
            heading=data.get("heading", None),
        )
        # particle_swap(self)

        # Calculate reward based on updated state & action
        control_heading = (
            data["heading"] if data.get("heading", None) else self.state.sensor_state[2]
        )
        control_delta_heading = (control_heading - self.state.sensor_state[2]) % 360
        reward = self.state.reward_func(
            state=None,
            action=(control_delta_heading, data.get("distance", 0)),
            particles=self.pf.particles,
        )

        belief_obs = self.env_observation()

        self.last_observation = observation

        return (belief_obs, reward, observation)

    def void_probability(self, actions, r_min, min_bound=0.8):

        particles = self.pf.particles
        p_outside_void = []
        for action in actions:
            particles = self.dynamics(particles, control=action)
            for t in range(self.state.n_targets):
                #print(particles[:,4*t] )
                B = 1-np.mean(particles[:,4*t] < r_min)
                p_outside_void.append(B)
                #print(f"probability outside void = {B}")
        
        if np.min(p_outside_void) >= min_bound:
            return True, particles
        return False, particles

    def rollout(self, actions):
        """Function to make n steps based on
           list of action indexes 

        Parameters
        ----------
        actions : array_like
            Actions to perform rollout

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

        particles = self.pf.particles
        for action in actions: 
            particles = self.dynamics(particles, control=action)
        return particles

    # returns observation, reward, done, info
    def step(self, action):
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
        #action = self.actions.index_to_action(action_idx)
        # Determine next state based on action & current state variables
        next_state = np.array(
            [
                self.state.update_state(target_state, action)
                for target_state in self.state.target_state
            ]
        )
        # Update absolute position of sensor
        self.state.update_sensor(action)
        # Get sensor observation
        observation = self.sensor.observation(next_state)
        # Update particle filter
        self.pf.update(np.array(observation), xp=self.pf.particles, control=action)
        particle_swap(self)
        # Calculate reward based on updated state & action
        reward = self.state.reward_func(
            state=next_state, action=action, particles=self.pf.particles
        )
        # reward = -1. * self.get_distance_error()
        # Update the state variables
        self.state.target_state = next_state

        env_obs = self.env_observation()
        self.iters += 1
        info = {"episode": {}}
        info["episode"]["l"] = self.iters
        info["episode"]["r"] = reward
        info["observation"] = observation

        return (env_obs, reward, 0, info)

    # def entropy_collision_reward(self, state, action_idx=None, delta=10, collision_weight=1):
    #     pf_r = self.pf.particles[:,0]
    #     pf_theta = np.radians(self.pf.particles[:,1])
    #     pf_x, pf_y = pol2cart(pf_r, pf_theta)
    #     xedges = np.arange(-150, 153, 3)
    #     yedges = np.arange(-150, 153, 3)
    #     b = np.histogram2d(pf_x, pf_y, bins=(xedges, yedges))
    #     b /= np.sum(b)
    #     b += 0.0000001

    #     H = -1. * np.sum([b * np.log(b)])
    #     collision_rate = np.mean(self.pf.particles[:,0] < delta)
    #     cost = H + collision_weight * collision_rate

    #     return -1. * cost

    def env_observation(self):
        """Helper function for environment observation

        Returns
        -------
        array_like
            Heatmap distribution of current observed particles
        """
        # return np.expand_dims(self.particle_heatmap_obs(self.pf.particles), axis=0)
        belief = self.pf.particles.reshape(
            len(self.pf.particles), self.state.n_targets, 4
        )
        # flattened pf map [2 x 100 x 100] -> [20000]
        pf_map = self.particle_heatmap_obs(belief).reshape(-1)
        mean_belief = []
        for t in range(self.state.n_targets):
            (
                _,
                _,
                _,
                _,
                mean_r,
                mean_theta,
                mean_heading,
                mean_spd,
            ) = particles_mean_belief(belief[:, t, :])
            mean_belief.extend([mean_r, mean_theta, mean_heading, mean_spd])
        # flattened mean belief [2 x 4] -> [8]
        mean_belief = np.array(mean_belief)

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
        heatmaps = []
        map_width = 600
        min_map = -1 * int(map_width / 2)
        max_map = int(map_width / 2)
        cell_size = 2  # (max_map - min_map)/max_map
        xedges = np.arange(min_map, max_map + cell_size, cell_size)
        yedges = np.arange(min_map, max_map + cell_size, cell_size)
        for t in range(self.state.n_targets):
            cart = np.array(
                list(map(pol2cart, belief[:, t, 0], np.radians(belief[:, t, 1])))
            )
            x = cart[:, 0]
            y = cart[:, 1]

            # Build two-dim histogram distribution
            h, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
            h = gaussian_filter(h, sigma=8)
            heatmaps.append(h)
        heatmaps = np.array(heatmaps)
        return heatmaps

    def get_absolute_particles(self):
        return np.array(
            [
                [
                    self.state.get_absolute_state(x[4 * t : 4 * (t + 1)])
                    for t in range(self.state.n_targets)
                ]
                for x in self.pf.particles
            ]
        )

    def get_absolute_target(self):
        return np.array(
            [self.state.get_absolute_state(state) for state in self.state.target_state]
        )

    def get_particle_centroids(self, particles=None):
        if particles is None: 
            particles = self.pf.particles
        centroids = []
        for t in range(self.state.n_targets):
            particles_x, particles_y = pol2cart(
                particles[:, 4 * t], np.radians(particles[:, (4 * t) + 1])
            )
            # centroid of particles x,y
            centroids.append([np.mean(particles_x), np.mean(particles_y)])

        return np.array(centroids)

    def get_particle_std_dev_cartesian(self, particles=None):
        if particles is None: 
            particles = self.pf.particles
        std_dev = []
        for t in range(self.state.n_targets):
            particles_x, particles_y = pol2cart(
                particles[:, 4 * t], np.radians(particles[:, (4 * t) + 1])
            )
            std_dev.append([np.std(particles_x), np.std(particles_y)])

        return np.array(std_dev)

    def get_particle_std_dev_polar(self, particles=None):
        if particles is None: 
            particles = self.pf.particles
        std_dev = []
        for t in range(self.state.n_targets):
            std_dev.append([np.std(particles[:, 4 * t]), np.std(particles[:, (4 * t) + 1])])

        return np.array(std_dev)

    def get_distance_error(self):
        mean_x, mean_y = self.get_particle_centroid()

        target_r = self.state.target_state[0]
        target_theta = np.radians(self.state.target_state[1])
        target_x, target_y = pol2cart(target_r, target_theta)

        centroid_distance_error = np.sqrt(
            (mean_x - target_x) ** 2 + (mean_y - target_y) ** 2
        )

        return centroid_distance_error


class RFEnv:
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
        self.pf = None

    def dynamics(self, particles, control=None, **kwargs):
        """Helper function for particle filter dynamics

        Returns
        -------
        array_like
            Updated particle state information
        """
        return np.array([list(self.state.update_state(p, control)) for p in particles])

    def reset(self, num_particles=2000):
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
            observe_fn=lambda states, **kwargs: np.array(
                [np.array(self.sensor.observation(x)) for x in states]
            ),
            n_particles=num_particles,
            dynamics_fn=self.dynamics,
            noise_fn=lambda x, **kwargs: x,
            resample_proportion=0.005,
            # noise_fn=lambda x:
            #            gaussian_noise(x, sigmas=[0.2, 0.2, 0.1, 0.05, 0.05]),
            weight_fn=lambda hyp, o, xp=None, **kwargs: [
                self.sensor.weight(None, o, state=x) for x in xp
            ],
            resample_fn=systematic_resample,
            column_names=["range", "heading", "relative_course", "own_speed"],
        )

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
        reward = self.state.reward_func(
            state=next_state, action_idx=action_idx, particles=self.pf.particles
        )
        # reward = -1. * self.get_distance_error()
        # Update the state variables
        self.state.target_state = next_state

        env_obs = self.env_observation()
        self.iters += 1
        info = {"episode": {}}
        info["episode"]["l"] = self.iters
        info["episode"]["r"] = reward
        info["observation"] = observation

        return (env_obs, reward, 0, info)

    def entropy_collision_reward(
        self, state, action_idx=None, delta=10, collision_weight=1
    ):
        pf_r = self.pf.particles[:, 0]
        pf_theta = np.radians(self.pf.particles[:, 1])
        pf_x, pf_y = pol2cart(pf_r, pf_theta)
        xedges = np.arange(-150, 153, 3)
        yedges = np.arange(-150, 153, 3)
        b = np.histogram2d(pf_x, pf_y, bins=(xedges, yedges))
        b /= np.sum(b)
        b += 0.0000001

        H = -1.0 * np.sum([b * np.log(b)])
        collision_rate = np.mean(self.pf.particles[:, 0] < delta)
        cost = H + collision_weight * collision_rate

        return -1.0 * cost

    def env_observation(self):
        """Helper function for environment observation

        Returns
        -------
        array_like
            Heatmap distribution of current observed particles
        """
        # return np.expand_dims(self.particle_heatmap_obs(self.pf.particles), axis=0)
        pf_map = np.expand_dims(
            self.particle_heatmap_obs(self.pf.particles), axis=0
        ).reshape(-1)
        _, _, _, _, mean_r, mean_theta, mean_heading, mean_spd = particles_mean_belief(
            self.pf.particles
        )
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
        cart = np.array(list(map(pol2cart, belief[:, 0], np.radians(belief[:, 1]))))
        x = cart[:, 0]
        y = cart[:, 1]

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

        particles_r = particles[:, 0]
        particles_theta = np.radians(particles[:, 1])
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

        centroid_distance_error = np.sqrt(
            (mean_x - target_x) ** 2 + (mean_y - target_y) ** 2
        )

        return centroid_distance_error
