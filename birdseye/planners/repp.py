import logging
import numpy as np 
from scipy.spatial import distance

from birdseye.utils import circ_tangents, cart2pol

class REPP:
    def __init__(self, env, min_std_dev, r_min, horizon, min_bound, target_selections):
        self.env = env
        self.min_std_dev = min_std_dev
        self.r_min = r_min
        self.horizon = horizon
        self.min_bound = min_bound
        self.target_selections = target_selections
    
    def get_action(self,):
    
        control_action = None
        std_dev = np.amax(self.env.get_particle_std_dev_cartesian(), axis=1) # get maximum standard deviation axis for each target
        #print(f"{std_dev=}")
        not_found = np.where(std_dev > self.min_std_dev)
        found = np.where(std_dev <= self.min_std_dev)
        target_selections_not_found = self.target_selections.copy()
        for f in found[0]: 
            target_selections_not_found.discard(f)
        if len(target_selections_not_found) == 0: 
            target_selections_not_found = {t for t in not_found[0]}
        std_dev_sorted = np.argsort(std_dev)
        not_found_sorted = np.argsort(std_dev[not_found])
        if len(not_found[0]):
            object_of_interest = not_found[0][np.argmin(std_dev[not_found])]
        else:
            print(f"All objects localised!")
            return None


        # get tangents and min distance point proposals
        #print(f"{env.get_particle_centroids()=}")
        centroids = self.env.get_particle_centroids()
        proposals = {}
        trajectories = {}
        #for i in target_selections_not_found: 
        #for i in range(env.state.n_targets):
        for i in self.target_selections:
            if self.env.state.n_targets > 1: 
                mean_other_centroids = [np.mean(np.delete(centroids,i,axis=0), axis=0)]
            else: 
                mean_other_centroids = [np.array([0,0])]

            target_proposals = circ_tangents([0,0], self.env.get_particle_centroids()[i], self.r_min)

            # for p in proposals: 
            #     print(f"{p=}")
            #     print(f"{mean_other_centroids[0]=}")
            #     print(f"{np.linalg.norm(np.array(p)-mean_other_centroids[0])=}")
            # print(f"{i=}, {mean_other_centroids=}")
            # print(f"{i=}, {proposals=}")
            if target_proposals is not None: 
                distances_to_other = distance.cdist(mean_other_centroids,target_proposals)[0]
                sorted_proposals = target_proposals[np.argsort(distances_to_other)]
                min_dist_proposal = target_proposals[np.argmin(distances_to_other)]
                proposals[i] = min_dist_proposal

                proposals[i] = cart2pol(min_dist_proposal[0],min_dist_proposal[1])
                trajectory = np.zeros((self.horizon, 2))
                trajectory[:,1] = np.minimum(proposals[i][0]/self.horizon, self.env.state.sensor_speed)
                trajectory[0,0] = np.degrees(proposals[i][1])
                trajectories[i] = trajectory

                void_condition, particles = self.env.void_probability(trajectory, self.r_min, min_bound=self.min_bound)
                # if void_condition and (i in target_selections):
                #     target_selections.remove(i)
                #     if len(target_selections) == 0: 
                #         target_selections = {t for t in range(env.state.n_targets)}
                if void_condition: 
                    #target_selections.discard(i)
                    self.target_selections.remove(i)
                    if len(self.target_selections) == 0: 
                        self.target_selections.update([t for t in range(self.env.state.n_targets)])
                    control_action = trajectory
                    #print(f"Found good trajectory!")
                    break
                    
        deg_width = 40 
        default_controls = np.linspace(-180,int(180-deg_width),int(360/deg_width))

        # if no optimal trajectory meets void constraint 
        if control_action is None:
            logging.info("Path planner (REPP): No optimal path could be calculated. Choosing from defaults.")
            # create trajectories from default controls 
            trajectories = []
            for c in default_controls:
                trajectory = np.zeros((self.horizon, 2))
                trajectory[:,1] = self.env.state.sensor_speed
                trajectory[0,0] = c
                trajectories.append(trajectory)

            # check void constraint for each trajectory 
            distances = []
            constrained_trajectories = []
            for trj in trajectories:
                void_condition, particles = self.env.void_probability(trj, self.r_min, min_bound=self.min_bound)
                if void_condition:
                    constrained_trajectories.append(trj)
                    distances.append(self.env.get_particle_centroids(particles=particles)[object_of_interest][0])
            # choose the sufficient trajectory that results in the min distance to centroid 
            if distances:
                control_action = constrained_trajectories[np.argmin(distances)]

        if control_action is None:
            logging.info(f"Path planner (REPP): No path satisfies void constraints. Choosing random path.")
            control_action = trajectories[np.random.randint(len(default_controls))]
            
        return control_action