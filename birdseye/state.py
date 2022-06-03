import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from .utils import pol2cart


class State:
    """Common base class for state & assoc methods
    """
    def __init__(self):
        pass

    def init_state(self):
        """Undefined initializing state method
        """
        pass

    def reward_func(self):
        """Undefined reward calc method
        """
        pass

    def update_state(self):
        """Undefined state updating method
        """
        pass

class RFMultiState(State):
    """RF Multi Target State
    """
    def __init__(self, n_targets=1, prob=0.9, target_speed=None, target_speed_range=None, target_movement=None, target_start=None, reward=None, simulated=True):

        self.state_dim = 4
        ### Target Settings
        # Transition probability
        self.prob_target_change_crs = prob
        # Target speed
        self.target_speed = float(target_speed) if target_speed is not None else 1.
        self.target_speed_range = [float(t) for t in target_speed_range.strip("'[]").split(',')] if target_speed_range is not None else [self.target_speed]
        # Target movement pattern
        self.target_movement = target_movement if target_movement is not None else 'random'
        self.target_move_iter = 0
        # Target start distance
        self.target_start = int(target_start) if target_start is not None else 150
        # Number of targets
        self.n_targets = int(n_targets) if n_targets is not None else 1

        ### Target & sensor states
        # Setup an initial random state
        self.target_state = None
        if simulated:
            self.target_state = self.init_target_state()
        # Setup an initial sensor state
        self.sensor_state = self.init_sensor_state()

        ### Reward
        # Setup reward
        reward = 'range_reward' if reward is None else reward
        self.AVAIL_REWARDS = {'range_reward' : self.range_reward,
                              'entropy_collision_reward': self.entropy_collision_reward,
                              'heuristic_reward': self.heuristic_reward,
                }
        self.reward_func = self.AVAIL_REWARDS[reward]
        if reward == 'range_reward':
            self.belief_mdp = False
        elif reward in ['entropy_collision_reward', 'heuristic_reward']:
            self.belief_mdp = True

    def __str__(self):
        print_str = ''
        print_str += 'RF Multi State Information\n'
        print_str += '# of targets: {}\n'.format(self.n_targets)
        print_str += 'Target states: {}\n'.format(self.target_state)
        print_str += 'Sensor state: {}\n'.format(self.sensor_state)
        return print_str

    def init_target_state(self):
        """Function to initialize a random state

        Returns
        -------
        array_like
            Randomly generated state variable array
        """
        # state is [range, bearing, relative course, own speed]
        #return np.array([random.randint(25,100), random.randint(0,359), random.randint(0,11)*30, self.target_speed])
        return [self.random_state() for _ in range(self.n_targets)]

    def init_particle_state(self):
        """Function to initialize a random particle state

        Returns
        -------
        array_like
            Randomly generated state variable array
        """
        # state is [range, bearing, relative course, own speed]
        #return np.array([random.randint(25,100), random.randint(0,359), random.randint(0,11)*30, self.target_speed])
        return [self.random_particle_state() for _ in range(self.n_targets)]

    def random_particle_state(self):
        """Function to initialize a random state

        Returns
        -------
        array_like
            Randomly generated state variable array
        """
        # state is [range, bearing, relative course, own speed]
        return np.array([random.randint(50,200), random.randint(0,359), random.randint(0,11)*30, random.randint(0,1)])

    def random_state(self):
        """Function to initialize a random state

        Returns
        -------
        array_like
            Randomly generated state variable array
        """
        # state is [range, bearing, relative course, own speed]
        return np.array([random.randint(50,self.target_start+25), random.randint(0,359), random.randint(0,11)*30, self.target_speed])


    def init_sensor_state(self):
        # state is [range, bearing, relative course, own speed]
        return np.array([0,0,0,0])

    # returns reward as a function of range, action, and action penalty or as a function of range only
    def heuristic_reward(self, state, action=None, action_idx=None, particles=None, action_penalty=-1.0, delta=20):
        """Function to calculate reward based on state and selected action

        Parameters
        ----------
        state : array_like
            List of current state variables
        action_idx : integer
            Index for action to make step
        action_penalty : float
            Penalty value to reward if action provided

        Returns
        -------
        reward_val : float
            Calculated reward value
        """

        # Set reward to 0/. as default
        reward_val = 0.
        if action is not None:
            if action[0] != 0:
                reward_val += action_penalty
        elif action_idx is not None: # returns reward as a function of range, action, and action penalty
            if action_idx not in [2,3]:
                reward_val += action_penalty

        col = 20
        lost = 150
        collision_rate = np.mean([np.mean(particles[:,4*t] < col) for t in range(self.n_targets)])
        lost_rate = np.mean([np.mean(particles[:,4*t] > lost) for t in range(self.n_targets)])
        collision_weight = -20
        lost_weight = -10
        reward_val += (collision_weight * collision_rate) + (lost_weight * lost_rate)

        return reward_val

    # returns reward as a function of range, action, and action penalty or as a function of range only
    def range_reward(self, state, action=None, action_idx=None, particles=None, action_penalty=-.05):
        """Function to calculate reward based on state and selected action

        Parameters
        ----------
        state : array_like
            List of current state variables
        action_idx : integer
            Index for action to make step
        action_penalty : float
            Penalty value to reward if action provided

        Returns
        -------
        reward_val : float
            Calculated reward value
        """

        # Set reward to 0/. as default
        reward_val = 0.
        state_ranges = [state[t,0] for t in range(self.n_targets)] #  state_range = state[0]
        max_state_range = np.max(state_ranges)
        min_state_range = np.min(state_ranges)

        if action is not None:
            if action[0] != 0:
                reward_val += action_penalty
        elif action_idx is not None: # returns reward as a function of range, action, and action penalty
            if action_idx not in [2,3]:
                reward_val += action_penalty


            if min_state_range >= 150:
                reward_val = -2 # reward to not lose track of contact
            elif min_state_range <= 15:
                reward_val = -20 # collision avoidance
            else:
                reward_val = 0.1 # being in "sweet spot" maximizes reward
        else: # returns reward as a function of range only
            if min_state_range >= 150:
                reward_val = -2 # reward to not lose track of contact
            elif min_state_range <= 15:
                reward_val = -200 # collision avoidance
            else:
                reward_val = 0.1
        return reward_val

    def entropy_collision_reward(self, state, action=None, action_idx=None, particles=None, delta=15, collision_weight=1):

        map_width = 600
        min_map = -1*int(map_width/2)
        max_map = int(map_width/2)
        cell_size = int((max_map - min_map)/max_map)
        cell_size = 2
        xedges = np.arange(min_map, max_map+cell_size, cell_size)
        yedges = np.arange(min_map, max_map+cell_size, cell_size)

        H = 0
        for t in range(self.n_targets):

            pf_r = particles[:,4*t]
            pf_theta = np.radians(particles[:,(4*t)+1])
            pf_x, pf_y = pol2cart(pf_r, pf_theta)
            b,_,_ = np.histogram2d(pf_x, pf_y, bins=(xedges, yedges))
            b = gaussian_filter(b, sigma=8)
            b += 0.0000001
            b /= np.sum(b)
            H += -1. * np.sum([b * np.log(b)])

        collision_rate = np.mean([np.mean(particles[:,4*t] < delta) for t in range(self.n_targets)])
        cost = H + collision_weight * collision_rate

        return -1. * cost


    # returns new state given last state and action (control)
    def update_state(self, state, control, target_update=False, transition_overwrite=None):
        """Update state based on state and action

        Parameters
        ----------
        state_vars : list
            List of current state variables
        control : action (tuple)
            Action tuple

        Returns
        -------
        State (array_like)
            Updated state values array
        """
        # Get current state vars
        r, theta, crs, spd = state
        control_spd = control[1]

        theta = theta % 360
        theta -= control[0]
        theta = theta % 360
        if theta < 0:
            theta += 360

        crs = crs % 360
        crs -= control[0]
        if crs < 0:
            crs += 360
        crs = crs % 360

        # Get cartesian coords
        x, y = pol2cart(r, np.radians(theta))

        # Generate next course given current course
        if target_update:
            spd = random.choice(self.target_speed_range)
            if self.target_movement == 'circular':
                d_crs, circ_spd = self.circular_control(50)
                crs += d_crs
                spd = circ_spd
            else:
                if random.random() >= self.prob_target_change_crs:
                    crs += random.choice([-1, 1]) * 30
        else:
            if random.random() >= self.prob_target_change_crs:
                crs += random.choice([-1, 1]) * 30
        crs %= 360
        if crs < 0:
            crs += 360

        spd = random.randint(0,1)

        # Transform changes to coords to cartesian
        dx, dy = pol2cart(spd, np.radians(crs))
        if transition_overwrite:
            dx, dy = transition_overwrite
        pos = [x + dx - control_spd, y + dy]

        r = np.sqrt(pos[0]**2 + pos[1]**2)
        theta_rad = np.arctan2(pos[1], pos[0])
        theta = np.degrees(theta_rad)
        if theta < 0:
            theta += 360

        return [r, theta, crs, spd]

    def update_sensor(self, control, bearing=None):
        r, theta_deg, crs, spd = self.sensor_state

        spd = control[1]

        crs = crs % 360
        crs += control[0]
        if bearing is not None:
            crs = bearing
        if crs < 0:
            crs += 360
        crs = crs % 360

        x, y = pol2cart(r, np.radians(theta_deg))

        dx, dy = pol2cart(spd, np.radians(crs))
        pos = [x + dx, y + dy]

        r = np.sqrt(pos[0]**2 + pos[1]**2)
        theta_deg = np.degrees(np.arctan2(pos[1], pos[0]))
        if theta_deg < 0:
            theta_deg += 360

        self.sensor_state = np.array([r, theta_deg, crs, spd])

    # returns absolute state given base state(absolute) and relative state
    def get_absolute_state(self, relative_state):
        r_t, theta_t, crs_t, spd = relative_state
        r_s, theta_s, crs_s, _ = self.sensor_state

        x_t, y_t = pol2cart(r_t, np.radians(theta_t+crs_s))
        x_s, y_s = pol2cart(r_s, np.radians(theta_s))

        x = x_t + x_s
        y = y_t + y_s
        r = np.sqrt(x**2 + y**2)
        theta_deg = np.degrees(np.arctan2(y, x))
        if theta_deg < 0:
            theta_deg += 360

        return [r, theta_deg, crs_s+crs_t, spd]

    def circular_control(self, size):
        self.target_move_iter += 1
        d_crs = 2*self.target_speed
        circ_spd = (6.5*size)/(360/self.target_speed)
        return [d_crs, circ_spd]

class RFState(State):
    """RF State
    """
    def __init__(self, prob=0.9, target_speed=None, target_speed_range=None, target_movement=None, target_start=None, reward=None):

        # Transition probability
        self.prob_target_change_crs = prob
        # Target speed
        self.target_speed = float(target_speed) if target_speed is not None else 1.
        self.target_speed_range = [float(t) for t in target_speed_range.strip("'[]").split(',')] if target_speed_range is not None else [self.target_speed]
        # Target movement pattern
        self.target_movement = target_movement if target_movement is not None else 'random'
        self.target_move_iter = 0
        # Target start distance
        self.target_start = int(target_start) if target_start is not None else 75
        # Setup an initial random state
        self.target_state = self.init_target_state()
        # Setup an initial sensor state
        self.sensor_state = self.init_sensor_state()
        # Setup reward
        reward = 'range_reward' if reward is None else reward
        self.AVAIL_REWARDS = {'range_reward' : self.range_reward,
                              'entropy_collision_reward': self.entropy_collision_reward,
                }
        self.reward_func = self.AVAIL_REWARDS[reward]
        if reward == 'range_reward':
            self.belief_mdp = False
        elif reward == 'entropy_collision_reward':
            self.belief_mdp = True


    def init_target_state(self):
        """Function to initialize a random state

        Returns
        -------
        array_like
            Randomly generated state variable array
        """
        # state is [range, bearing, relative course, own speed]
        #return np.array([random.randint(25,100), random.randint(0,359), random.randint(0,11)*30, self.target_speed])
        return np.array([random.randint(self.target_start-25,self.target_start+25), random.randint(0,359), random.randint(0,11)*30, self.target_speed])

    def random_state(self):
        """Function to initialize a random state

        Returns
        -------
        array_like
            Randomly generated state variable array
        """
        # state is [range, bearing, relative course, own speed]
        return np.array([random.randint(10,200), random.randint(0,359), random.randint(0,11)*30, self.target_speed])


    def init_sensor_state(self):
        # state is [range, bearing, relative course, own speed]
        return np.array([0,0,0,0])

    # returns reward as a function of range, action, and action penalty or as a function of range only
    def range_reward(self, state, action_idx=None, particles=None, action_penalty=-.05):
        """Function to calculate reward based on state and selected action

        Parameters
        ----------
        state : array_like
            List of current state variables
        action_idx : integer
            Index for action to make step
        action_penalty : float
            Penalty value to reward if action provided

        Returns
        -------
        reward_val : float
            Calculated reward value
        """

        # Set reward to 0/. as default
        reward_val = 0.
        state_range = state[0]

        if action_idx is not None: # returns reward as a function of range, action, and action penalty
            if (1 < action_idx < 4):
                action_penalty = 0

            if state_range >= 150:
                reward_val = -2 + action_penalty # reward to not lose track of contact
            elif state_range <= 10:
                reward_val = -2 + action_penalty # collision avoidance
            else:
                reward_val = 0.1 + action_penalty # being in "sweet spot" maximizes reward
        else: # returns reward as a function of range only
            if state_range >= 150:
                reward_val = -2 # reward to not lose track of contact
            elif state_range <= 10:
                reward_val = -200 # collision avoidance
            else:
                reward_val = 0.1
        return reward_val

    def entropy_collision_reward(self, state, action_idx=None, particles=None, delta=10, collision_weight=1):
        pf_r = particles[:,0]
        pf_theta = np.radians(particles[:,1])
        pf_x, pf_y = pol2cart(pf_r, pf_theta)
        xedges = np.arange(-150, 153, 3)
        yedges = np.arange(-150, 153, 3)
        b,_,_ = np.histogram2d(pf_x, pf_y, bins=(xedges, yedges))

        b += 0.0000001
        b /= np.sum(b)
        H = -1. * np.sum([b * np.log(b)])
        collision_rate = np.mean(particles[:,0] < delta)
        cost = H + collision_weight * collision_rate

        return -1. * cost


    # returns new state given last state and action (control)
    def update_state(self, state, control, target_update=False):
        """Update state based on state and action

        Parameters
        ----------
        state_vars : list
            List of current state variables
        control : action (tuple)
            Action tuple

        Returns
        -------
        State (array_like)
            Updated state values array
        """
        # Get current state vars
        r, theta, crs, spd = state
        control_spd = control[1]

        theta = theta % 360
        theta -= control[0]
        theta = theta % 360
        if theta < 0:
            theta += 360

        crs = crs % 360
        crs -= control[0]
        if crs < 0:
            crs += 360
        crs = crs % 360

        # Get cartesian coords
        x, y = pol2cart(r, np.radians(theta))

        # Generate next course given current course
        if target_update:
            spd = random.choice(self.target_speed_range)
            if self.target_movement == 'circular':
                d_crs, circ_spd = self.circular_control(50)
                crs += d_crs
                spd = circ_spd
            else:
                if random.random() >= self.prob_target_change_crs:
                    crs += random.choice([-1, 1]) * 30
        else:
            if random.random() >= self.prob_target_change_crs:
                crs += random.choice([-1, 1]) * 30
        crs %= 360
        if crs < 0:
            crs += 360

        # Transform changes to coords to cartesian
        dx, dy = pol2cart(spd, np.radians(crs))
        pos = [x + dx - control_spd, y + dy]

        r = np.sqrt(pos[0]**2 + pos[1]**2)
        theta_rad = np.arctan2(pos[1], pos[0])
        theta = np.degrees(theta_rad)
        if theta < 0:
            theta += 360

        return (r, theta, crs, spd)

    def update_sensor(self, control):
        r, theta_deg, crs, spd = self.sensor_state

        spd = control[1]

        crs = crs % 360
        crs += control[0]
        if crs < 0:
            crs += 360
        crs = crs % 360

        x, y = pol2cart(r, np.radians(theta_deg))

        dx, dy = pol2cart(spd, np.radians(crs))
        pos = [x + dx, y + dy]

        r = np.sqrt(pos[0]**2 + pos[1]**2)
        theta_deg = np.degrees(np.arctan2(pos[1], pos[0]))
        if theta_deg < 0:
            theta_deg += 360

        self.sensor_state = np.array([r, theta_deg, crs, spd])

    # returns absolute state given base state(absolute) and relative state
    def get_absolute_state(self, relative_state):
        r_t, theta_t, crs_t, spd = relative_state
        r_s, theta_s, crs_s, _ = self.sensor_state

        x_t, y_t = pol2cart(r_t, np.radians(theta_t+crs_s))
        x_s, y_s = pol2cart(r_s, np.radians(theta_s))

        x = x_t + x_s
        y = y_t + y_s
        r = np.sqrt(x**2 + y**2)
        theta_deg = np.degrees(np.arctan2(y, x))
        if theta_deg < 0:
            theta_deg += 360

        return [r, theta_deg, crs_s+crs_t, spd]

    def circular_control(self, size):
        self.target_move_iter += 1
        d_crs = 2*self.target_speed
        circ_spd = (6.5*size)/(360/self.target_speed)
        return [d_crs, circ_spd]


AVAIL_STATES = {
     # 'rfstate' : RFState,
    'rfmultistate' : RFMultiState
}

def get_state(state_name=''):
    """Convenience function for retrieving BirdsEye state methods
    Parameters
    ----------
    state_name : {'rfstate'}
        Name of state method.
    Returns
    -------
    state_obj : State class object
        BirdsEye state method.
    """
    state_name = state_name.lower()
    if state_name in AVAIL_STATES:
        state_obj = AVAIL_STATES[state_name]
        return state_obj
    else:
        raise ValueError('Invalid action method name, {}, entered. Must be '
                         'in {}'.format(state_name, AVAIL_STATES.keys()))

