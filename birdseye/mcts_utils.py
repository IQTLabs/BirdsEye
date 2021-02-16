# mcts_utils.jl


# Imports
import random
import numpy as np
from pfilter import ParticleFilter, systematic_resample
import matplotlib.pyplot as plt
from IPython.display import clear_output


######################################
### generative model
######################################
# generate next course given current course
def next_crs(crs):
    if random.random() < 0.9:
        return crs
    crs = (crs + random.choice([-1,1])*30) % 360
    if crs < 0:
        crs += 360
    return crs


# alternate next course-- uses different probability to represent model error
def next_crs_gen(crs):
    if random.random() < 0.75:
        return crs
    crs = (crs + random.choice([-1,1])*30) % 360
    if crs < 0:
        crs += 360
    return crs


# returns new state given last state and action (control)
def f(state, control):
    TGT_SPD = 1
    r, theta, crs, spd = state
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

    x = r*np.cos(np.pi / 180 * theta)
    y = r*np.sin(np.pi / 180 * theta)

    pos = [x + TGT_SPD * np.cos(np.pi / 180 * crs) - spd, y + TGT_SPD * np.sin(np.pi / 180 * crs)]
    crs = next_crs(crs)

    r = np.sqrt(pos[0]**2 + pos[1]**2)
    theta = np.arctan2(pos[1], pos[0]) * 180 / np.pi
    if theta < 0:
        theta += 360
    return (r, theta, crs, spd)


#same as f, except returns a list
def f2(x, u):

    return list(f(x,u))


ACTION_PENALTY = -.05


# returns reward as a function of range, action, and action penalty or as a function of range only
def r(s, u=None, action_penalty=ACTION_PENALTY):
    state_range = s[0]

    if u is not None: # returns reward as a function of range, action, and action penalty
        if (2 < u < 5):
            action_penalty = 0
        if state_range >= 150:
            return (-2 + action_penalty) # reward to not lose track of contact
        if state_range <= 10:
            return (-2 + action_penalty) # collision avoidance
        return (0.1 + action_penalty) # being in "sweet spot" maximizes reward
    else: # returns reward as a function of range only
        if state_range >= 150:
            return -2 # reward to not lose track of contact
        if state_range <= 10:
            return -200 # collision avoidance
        return 0.1 # being in "sweet spot" maximizes reward



# Action space and function to convert from action to index and vice versa
action_space = ((-30,1), (-30, 2), (0, 1), (0, 2), (30, 1), (30, 2))


#returns action given an index
def action_to_index(a):
    return int(np.trunc(2*(a[0] / 30 + 1) + a[1]))-1


#function version of action_space
def actions():
    return ((-30,1), (-30, 2), (0, 1), (0, 2), (30, 1), (30, 2))


#returns index of action given an action
def index_to_action(a):
    a = a + 1
    if a % 2 == 0:
        return (int(np.trunc((((a - 2) / 2) - 1) * 30)), 2)
    else:
        return (int(np.trunc((((a - 1) / 2) - 1) * 30)), 1)



##################################################################
# MCTS Algorithm
##################################################################

def arg_max_action(Q, N, history, c=None, exploration_bonus=False):

    # only need to compute if exploration possibility
    if exploration_bonus:
        N_h = 0
        for action in list(map(action_to_index, action_space)):
            new_index = history.copy()
            new_index.append(action)
            N_h += N[tuple(new_index)]

    values = []
    for action in list(map(action_to_index, action_space)):
        new_index = history.copy()
        new_index.append(action)

        # best action with exploration possibility
        if exploration_bonus:
            if N[tuple(new_index)] == 0:
                return action

            # compute exploration bonus, checking for zeroes (I don't think this will ever occur anyway...)
            if np.log(N_h) < 0:
                numerator = 0
            else:
                numerator = np.sqrt(np.log(N_h))

            denominator = N[tuple(new_index)]
            exp_bonus = c * numerator / denominator
            values.append(Q[tuple(new_index)] + exp_bonus)
        else:
            values.append(Q[tuple(new_index)])
    return np.argmax(values)


##################################################################
# Rollout
##################################################################
def rollout_random(state, depth):

    if depth == 0:
        return 0

    # random action
    random_action_index = random.choice(list(map(action_to_index, action_space)))
    action = index_to_action(random_action_index)

    # generate next state and reward with random action; observation doesn't matter
    state_prime = f2(state, action)
    reward = r(tuple(state_prime), action_to_index(action))

    return reward + lambda_arg * rollout_random(state_prime, depth-1)


##################################################################
# Simulate
##################################################################
def simulate(sensor, Q, N, state, history, depth, c):

    if depth == 0:
        return (Q, N, 0)

    # expansion
    test_index = history.copy()
    test_index.append(1)

    if tuple(test_index) not in Q:

        for action in list(map(action_to_index, action_space)):
            # initialize Q and N to zeros
            new_index = history.copy()
            new_index.append(action)
            Q[tuple(new_index)] = 0
            N[tuple(new_index)] = 0

        # rollout
        return (Q, N, rollout_random(state, depth))

    # search
    # find optimal action to explore
    search_action_index = arg_max_action(Q, N, history, c, True)
    action = index_to_action(search_action_index)

    # take action; get new state, observation, and reward
    state_prime = f2(state, action)
    observation = sensor.observation(state_prime)
    reward = r(tuple(state_prime), action_to_index(action))

    # recursive call after taking action and getting observation
    new_history = history.copy()
    new_history.append(search_action_index)
    new_history.append(observation)
    (Q, N, successor_reward) = simulate(sensor, Q, N, state_prime, new_history, depth-1, c)
    q = reward + lambda_arg * successor_reward

    # update counts and values
    update_index = history.copy()
    update_index.append(search_action_index)
    N[tuple(update_index)] += 1
    Q[tuple(update_index)] += ((q - Q[tuple(update_index)]) / N[tuple(update_index)])

    return (Q, N, q)


##################################################################
# Select Action
##################################################################
def select_action(sensor, Q, N, belief, depth, c, iterations):

    # empty history at top recursive call
    history = []

    # loop
    # timed loop, how long should intervals be?
    #counter = 0
    #start_time = time_ns()
    #while (time_ns() - start_time) / 1.0e9 < 1 # 1 second timer to start

    # number of iterations
    counter = 0
    while counter < iterations:

        # draw state randomly based on belief state (pick a random particle)
        state = random.choice(belief)

        # simulate
        simulate(sensor, Q, N, state.astype(float), history, depth, c)

        counter+=1


    best_action_index = arg_max_action(Q, N, history)
    action = index_to_action(best_action_index)
    return (Q, N, action)

def dynamics(particles, control=None, **kwargs):
    return np.array([f2(p, control) for p in particles])

def random_state():
    return np.array([random.randint(25,100), random.randint(0,359), random.randint(0,11)*30, 1])

def near_state(sensor, state): 
    return np.array(sensor.gen_state(sensor.observation(state)))

##################################################################
# Trial
##################################################################
lambda_arg = 0.95
def mcts_trial(sensor, depth, c, plotting=False, num_particles=500, iterations=1000, fig=None, ax=None):


    # Initialize true state and belief state (particle filter); we assume perfect knowledge at start of simulation (could experiment otherwise with random beliefs)
    #num_particles = 500

    # true state
    # state is [range, bearing, relative course, own speed]
    # assume a starting position within range of sensor and not too close
    true_state = np.array([random.randint(25,100), random.randint(0,359), random.randint(0,11)*30, 1])
    pf = ParticleFilter(
                        prior_fn=lambda n: np.array([near_state(sensor, true_state) for i in range(n)]),
                        observe_fn=lambda states, **kwargs: np.array([np.array(sensor.observation(x)) for x in states]),
                        n_particles=num_particles,
                        #dynamics_fn=lambda particles, **kwargs: [f2(p, control) for p in particles],
                        dynamics_fn=dynamics,
                        noise_fn=lambda x, **kwargs: x,
                        #noise_fn=lambda x:
                        #            gaussian_noise(x, sigmas=[0.2, 0.2, 0.1, 0.05, 0.05]),
                        weight_fn=lambda hyp, o, xp=None,**kwargs: [sensor.weight(None, o, xp=x) for x in xp],
                        resample_fn=systematic_resample,
                        column_names = ['range', 'bearing', 'relative_course', 'own_speed'])

    # assuming image of the same dimensions/type as blob will produce
    # pf.update(image)
    ##model = ParticleFilterModel{Vector{Float64}}(f2, g)
    ##pfilter = SIRParticleFilter(model, num_particles)



    # belief state
    # assume perfect knowledge at first time step
    ##belief = ParticleCollection([true_state for i in 1:num_particles])
    belief = pf.particles
        # Simulation prep/initialization; for now we start with no prior knowledge for Q values/N values, could incorporate this later

    #build_plot(true_state, belief)

    # global Q and N dictionaries, indexed by history (and optionally action to follow all in same array; using ints)
    ##Q = Dict{Array{Int64,1},Float64}()
    Q = {}
    ##N = Dict{Array{Int64,1},Float64}()
    N = {}

    # not manipulating these parameters for now, in global scope
    # lambda, discount factor
    ##lambda = 0.95

    # experimenting with different parameter values
    # experiment with different depth parameters
    depth = depth
    # exploration factor, experiment with different values
    c = c

    # don't need to modify history tree at first time step
    action = None
    observation = None

    # run simulation

    total_reward = 0
    total_col = 0
    total_loss = 0

    # 500 time steps with an action to be selected at each
    num_iters = 500
    plots = []
    for time_step in range(num_iters):

        #if time_step % 100 == 0
        #    @show time_step
        #end

        # NOTE: we found restarting history tree at each time step yielded better results
        # if action taken, modify history tree
        if action is not None:
            #(Q,N) = modify_history_tree(Q, N, action, observation)
            Q = {}
            N = {}

        # select an action
        (Q, N, action) = select_action(sensor, Q, N, belief, depth, c, iterations)

        # take action; get next true state, obs, and reward
        next_state = f2(true_state, action)
        observation = sensor.observation(next_state)
        #print('true_state = {}, next_state = {}, action = {}, observation = {}'.format(true_state, next_state, action, observation))
        reward = r(tuple(next_state), action_to_index(action))
        true_state = next_state

        # update belief state (particle filter)
        #belief = update(pfilter, belief, action, observation)
        pf.update(np.array(observation), xp=belief, control=action)
        belief = pf.particles

        # accumulate reward
        total_reward += reward
        if true_state[0] < 10:
            total_col = 1

        if plotting:
            #push!(plots, build_plot(true_state, belief))
            build_plot(true_state, belief, fig, ax, time_step)


        # TODO: flags for collision, lost track, end of simulation lost track



    if true_state[0] > 150:
        total_loss = 1

    return (total_reward, plots, total_col, total_loss)


##################################################################
# Plotting
##################################################################

def build_plot(xp, b, fig=None, ax=None, time_step=None):
    grid_r, grid_theta = [],[]
    plot_r = [row[0] for row in b]
    plot_theta = np.array([row[1] for row in b])*np.pi/180
    plot_x_theta = xp[1]*np.pi/180
    plot_x_r = xp[0]

    
    #plt.figure(1)
    clear_output(wait=True)
    #plt.figure()
    plt.polar(plot_theta, plot_r, 'ro')
    plt.polar(plot_x_theta, plot_x_r, 'bo')
    plt.ylim(-150,150)
    plt.title('iteration {}'.format(time_step))
    
    #ax.clear()
    #ax.plot(plot_theta, plot_r, 'ro')
    #ax.plot(plot_x_theta, plot_x_r, 'bo')
    plt.show()
    #fig.canvas.draw()
    #fig.clf()
    #plt.pause(0.0001)
    #plt.clf()
    



