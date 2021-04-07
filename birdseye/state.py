import random
import numpy as np 
from .utils import pol2cart


class State(object):
    """Common base class for state & assoc methods
    """
    def __init__(self, simulated=False):
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


class RFState(State):
    """RF State
    """
    def __init__(self, prob=0.9, simulated=False):
        # Flag for simulation vs real data
        self.simulated = simulated
        # Transition probability
        self.prob = prob
        # Setup an initial random state
        self.target_state = self.init_target_state()
        # Setup an initial sensor state 
        self.sensor_state = self.init_sensor_state()
    

    def init_target_state(self):
        # state is [range, bearing, relative course, own speed]
        return np.array([random.randint(25,100), random.randint(0,359), random.randint(0,11)*30, 1])
   
    def init_sensor_state(self): 
        # state is [range, bearing, relative course, own speed]
        return np.array([0,0,0,0])

    # returns reward as a function of range, action, and action penalty or as a function of range only
    def reward_func(self, state, action_idx=None, action_penalty=-.05):
    
        # Set reward to 0/. as default
        reward_val = 0.
        state_range = state[0]

        if action_idx is not None: # returns reward as a function of range, action, and action penalty
            if (2 < action_idx < 5):
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


    # returns new state given last state and action (control)
    def update_state(self, target_state, control):
        TGT_SPD = 1
        r, theta, crs, spd = target_state
        spd = control[1]
        
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
        
        x, y = pol2cart(r, np.radians(theta))

        dx, dy = pol2cart(TGT_SPD, np.radians(crs))
        pos = [x + dx - spd, y + dy]

        # generate next course given current course
        if random.random() >= self.prob:
            crs += random.choice([-1, 1]) * 30
            crs %= 360
            if crs < 0:
                crs += 360

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

        x_t, y_t = pol2cart(r_t, np.radians(theta_t))
        x_s, y_s = pol2cart(r_s, np.radians(theta_s))

        x = x_t + x_s
        y = y_t + y_s 
        r = np.sqrt(x**2 + y**2)
        theta_deg = np.degrees(np.arctan2(y, x))
        if theta_deg < 0:
            theta_deg += 360

        return [r, theta_deg, crs_s+crs_t, spd]

    


