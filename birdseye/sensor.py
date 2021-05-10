import random
import numpy as np 
import scipy.stats

class Sensor(object):
    """Common base class for sensor & assoc methods
    """
    def __init__(self):
        pass

    def observation(self, state):
        """Undefined observation sample method
        """
        pass

    def weight(self, hyp, obs, state):
        """Undefined method for importance
           weight of a state given observation
        """
        pass

    def acceptance(self, state):
        """Undefined method for defining
           detector acceptance pattern
        """
        pass

class SignalStrength(Sensor): 
    """
    Uses signal strength as observation
    """
    def __init__(self): 
        self.num_avail_obs = 1 
        self.std_dev = 10

    def weight(self, hyp, obs, state): 
        expected_r = state[0]
        obs_r = np.sqrt(1/obs)
        # Gaussian weighting function
        numer_fact = np.power(expected_r - obs_r, 2.) 
        denom_fact = 2 * np.power(self.std_dev, 2.)
        weight = np.exp( - numer_fact / denom_fact)
        return weight
    
    # samples observation given state
    def observation(self, state):
        return 1/ ((np.random.normal(state[0], self.std_dev)) ** 2)

    # sample state from observation 
    def gen_state(self, obs):
        r_dist = np.sqrt(1/obs)    
        return [r_dist, random.randint(0,359), random.randint(0,11)*30, 1]

    def near_state(self, state): 
        return np.array(self.gen_state(self.observation(state)))
    

class Drone(Sensor): 
    """Drone sensor
    """
    def __init__(self): 
        self.num_avail_obs = 2

    # importance weight of state given observation
    def weight(self, hyp, obs, state):

        # Get acceptance value for state value
        obs_weight = self.acceptance(state)
        
        # Convolve acceptance with observation weight 
        if obs == 1: 
            obs_weight *= self.obs1_prob(state)
        elif obs == 0: 
            obs_weight *= 1-self.obs1_prob(state)
        else:
            raise ValueError("Observation number ({}) outside acceptable int values: 0-{}"\
                             .format(obs, self.num_avail_obs-1))

        return obs_weight

    def acceptance(self, state):
        return 1.

    # samples observation given state
    def observation(self, state):

        obs1_val = self.obs1_prob(state)
        weights = [1.-obs1_val, obs1_val]
        obsers = [0, 1]
        return random.choices(obsers, weights)[0]

    # probability of observation 1 
    def obs1_prob(self, state): 
        rel_bearing = state[1]
        if -60 <= rel_bearing <= 60: 
            return 0.9
        elif 120 <= rel_bearing <= 240: 
            return 0.1
        else: 
            return 0.5 
            
    # sample state from observation 
    def gen_state(self, obs): 
        
        if obs == 1: 
            bearing = random.randint(-60,60)
        elif obs == 0: 
            bearing = random.randint(120, 240)
        
        if bearing < 0: 
            bearing += 360
            
        return [random.randint(25,100), bearing, random.randint(0,11)*30, 1]

    def near_state(self, state): 
        return np.array(self.gen_state(self.observation(state)))


class Bearing(Sensor): 
    def __init__(self, sensor_range = 150): 
        self.sensor_range = sensor_range
        self.num_avail_obs = 4

    # importance weight of state given observation
    def weight(self, hyp, obs, state):

        # Get acceptance value for state value
        obs_weight = self.acceptance(state)

        # Convolve acceptance with observation weight 
        if obs == 0:
            obs_weight *= self.obs0(state)
        elif obs == 1:
            obs_weight *= self.obs1(state)
        elif obs == 2:
            obs_weight *= self.obs2(state)
        elif obs == 3:
            obs_weight *= self.obs3(state)
        else:
            raise ValueError("Observation number ({}) outside acceptable int values: 0-{}"\
                             .format(obs, self.num_avail_obs-1))

        return obs_weight

    def acceptance(self, state):
        return 1.

    # samples observation given state
    def observation(self, state):
        weights = [self.obs0(state), self.obs1(state), self.obs2(state), self.obs3(state)]
        obsers = [0, 1, 2, 3]
        return random.choices(obsers, weights)[0]

    # sample state from observation 
    def gen_state(self, obs): 
        
        if obs == 0: 
            bearing = random.randint(-60,60)
        elif obs == 1: 
            bearing = random.choice([random.randint(60,90), random.randint(270, 300)])
        elif obs == 2: 
            bearing = random.choice([random.randint(90,120), random.randint(240,270)])
        elif obs == 3: 
            bearing = random.randint(120, 240)
        
        if bearing < 0: 
            bearing += 360
            
        return [random.randint(25,100), bearing, random.randint(0,11)*30, 1]

    def obs1(self, state):
        #rel_brg = state[1] - state[3]
        rel_brg = state[1]
        state_range = state[0]
        if rel_brg < 0:
            rel_brg += 360
        if ((60 < rel_brg < 90) or (270 < rel_brg < 300)) and (state_range < self.sensor_range/2):
            return 1
        elif ((60 < rel_brg < 90) or (270 < rel_brg < 300)) and (state_range < self.sensor_range):
            return 2-2*state_range/self.sensor_range
        else:
            return 0.0001

    def obs2(self, state):
        #rel_brg = state[1] - state[3]
        rel_brg = state[2]
        state_range = state[1]
        if rel_brg < 0:
            rel_brg += 360
        if ((90 <= rel_brg < 120) or (240 < rel_brg <= 270)) and (state_range < self.sensor_range/2):
            return 1
        elif ((90 <= rel_brg < 120) or (240 < rel_brg <= 270)) and (state_range < self.sensor_range):
            return 2-2*state_range/self.sensor_range
        else:
            return 0.0001

    def obs3(self, state):
        #rel_brg = state[1] - state[3]
        rel_brg = state[1]
        state_range = state[0]
        if rel_brg < 0:
            rel_brg += 360
        if (120 <= rel_brg <= 240) and (state_range < self.sensor_range/2):
            return 1
        elif (120 <= rel_brg <= 240) and (state_range < self.sensor_range):
            return 2-2*state_range/self.sensor_range
        else:
            return 0.0001

    def obs0(self, state):
        #rel_brg = state[1] - state[3]
        rel_brg = state[1]
        state_range = state[0]
        if rel_brg < 0:
            rel_brg += 360
        if (rel_brg <= 60) or (rel_brg >=300) or (state_range >= self.sensor_range):
            return 1
        if (not(self.obs1(state) > 0) and not(self.obs2(state) > 0) and not(self.obs3(state) > 0)):
            return 1
        elif (120 <= rel_brg <= 240) and (self.sensor_range/2 < state_range < self.sensor_range):
            return 2*state_range/self.sensor_range - 1
        elif ((90 <= rel_brg < 120) or (240 < rel_brg <= 270)) and (self.sensor_range/2 < state_range < self.sensor_range):
            return 2*state_range/self.sensor_range -1
        elif ((60 <= rel_brg < 90) or (270 < rel_brg <= 300)) and (self.sensor_range/2 < state_range < self.sensor_range):
            return 2*state_range/self.sensor_range - 1
        else:
            return 0.0001


AVAIL_SENSORS = {'drone' : Drone,
                 'bearing' : Bearing,
                 'signalstrength': SignalStrength
                }

def get_sensor(sensor_name=''):
    """Convenience function for retrieving BirdsEye sensor methods
    Parameters
    ----------
    sensor_name : {'simpleactions'}
        Name of sensor method.
    Returns
    -------
    sensor_obj : Sensor class object
        BirdsEye sensor method.
    """
    if sensor_name in AVAIL_SENSORS:
        sensor_obj = AVAIL_SENSORS[sensor_name]
        return sensor_obj
    else:
        raise ValueError('Invalid sensor method name, {}, entered. Must be '
                         'in {}'.format(sensor_name, AVAIL_SENSORS.keys()))

