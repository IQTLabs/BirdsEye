
import random

SENSOR_RANGE = 150

def obs1(state):
    #rel_brg = state[1] - state[3]
    rel_brg = state[1]
    state_range = state[0]
    if rel_brg < 0:
        rel_brg += 360
    if ((60 < rel_brg < 90) or (270 < rel_brg < 300)) and (state_range < SENSOR_RANGE/2):
        return 1
    elif ((60 < rel_brg < 90) or (270 < rel_brg < 300)) and (state_range < SENSOR_RANGE):
        return 2-2*state_range/SENSOR_RANGE
    else:
        return 0.0001

def obs2(state):
    #rel_brg = state[1] - state[3]
    rel_brg = state[2]
    state_range = state[1]
    if rel_brg < 0:
        rel_brg += 360
    if ((90 <= rel_brg < 120) or (240 < rel_brg <= 270)) and (state_range < SENSOR_RANGE/2):
        return 1
    elif ((90 <= rel_brg < 120) or (240 < rel_brg <= 270)) and (state_range < SENSOR_RANGE):
        return 2-2*state_range/SENSOR_RANGE
    else:
        return 0.0001

def obs3(state):
    #rel_brg = state[1] - state[3]
    rel_brg = state[1]
    state_range = state[0]
    if rel_brg < 0:
        rel_brg += 360
    if (120 <= rel_brg <= 240) and (state_range < SENSOR_RANGE/2):
        return 1
    elif (120 <= rel_brg <= 240) and (state_range < SENSOR_RANGE):
        return 2-2*state_range/SENSOR_RANGE
    else:
        return 0.0001

def obs0(state):
    #rel_brg = state[1] - state[3]
    rel_brg = state[1]
    state_range = state[0]
    if rel_brg < 0:
        rel_brg += 360
    if (rel_brg <= 60) or (rel_brg >=300) or (state_range >= SENSOR_RANGE):
        return 1
    if (not(obs1(state) > 0) and not(obs2(state) > 0) and not(obs3(state) > 0)):
        return 1
    elif (120 <= rel_brg <= 240) and (SENSOR_RANGE/2 < state_range < SENSOR_RANGE):
        return 2*state_range/SENSOR_RANGE - 1
    elif ((90 <= rel_brg < 120) or (240 < rel_brg <= 270)) and (SENSOR_RANGE/2 < state_range < SENSOR_RANGE):
        return 2*state_range/SENSOR_RANGE -1
    elif ((60 <= rel_brg < 90) or (270 < rel_brg <= 300)) and (SENSOR_RANGE/2 < state_range < SENSOR_RANGE):
        return 2*state_range/SENSOR_RANGE - 1
    else:
        return 0.0001

# returns observation given state
def g(x, a, xp, o):
    if o == 0:
        return obs0(xp)
    if o == 1:
        return obs1(xp)
    if o == 2:
        return obs2(xp)
    if o == 3:
        return obs3(xp)

# weight_fn
def weight(hyp, o, xp):
    if o == 0:
        return obs0(xp)
    if o == 1:
        return obs1(xp)
    if o == 2:
        return obs2(xp)
    if o == 3:
        return obs3(xp)

# samples observation given state
def h(x):
    weights = [obs0(x), obs1(x), obs2(x), obs3(x)]
    obsers = [0, 1, 2, 3]
    return random.choices(obsers, weights)[0]

# sample state from observation 
def gen_state(obs): 
    
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
