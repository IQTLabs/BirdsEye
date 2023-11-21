import numpy as np

from birdseye.utils import circ_tangents, cart2pol


class LAVAPilot:
    def __init__(self, env, min_std_dev, r_min, horizon, min_bound):
        self.env = env
        self.min_std_dev = min_std_dev
        self.r_min = r_min
        self.horizon = horizon
        self.min_bound = min_bound

    def get_action(
        self,
    ):
        control_action = None
        std_dev = np.amax(
            self.env.get_particle_std_dev_cartesian(), axis=1
        )  # get maximum standard deviation axis for each target
        # print(f"{std_dev=}")
        not_found = np.where(std_dev > self.min_std_dev)
        if len(not_found[0]):
            object_of_interest = np.argmin(std_dev[not_found])
        else:
            print(f"All objects localised!")
            return None
        # print(f"{object_of_interest=}")
        # get tangents and min distance point proposals
        proposals = circ_tangents(
            [0, 0], self.env.get_particle_centroids()[object_of_interest], self.r_min
        )
        # print(f"{env.get_particle_centroids()[object_of_interest]=}")
        if proposals is not None:
            # convert proposal end points to polar coordinates
            proposals = [cart2pol(p[0], p[1]) for p in proposals]
            # print(f"{proposals=}")

            # create trajectories of length=horizon from proposal end points (first turn, then move straight)
            trajectories = []
            for p in proposals:
                trajectory = np.zeros((self.horizon, 2))
                trajectory[:, 1] = np.minimum(
                    p[0] / self.horizon, self.env.state.sensor_speed
                )
                trajectory[0, 0] = np.degrees(p[1])
                trajectories.append(trajectory)

            # check void probability contstraint for each trajectory, choose first that is sufficient
            for trj in trajectories:
                void_condition, particles = self.env.void_probability(
                    trj, self.r_min, min_bound=self.min_bound
                )
                if void_condition:
                    control_action = trj
                    # print(f"Found good trajectory!")
                    break

        deg_width = 40
        default_controls = np.linspace(-180, int(180 - deg_width), int(360 / deg_width))

        # if no optimal trajectory meets void constraint
        if control_action is None:
            # create trajectories from default controls
            trajectories = []
            for c in default_controls:
                trajectory = np.zeros((self.horizon, 2))
                trajectory[:, 1] = self.env.state.sensor_speed
                trajectory[0, 0] = c
                trajectories.append(trajectory)

            # check void constraint for each trajectory
            distances = []
            constrained_trajectories = []
            for trj in trajectories:
                void_condition, particles = self.env.void_probability(
                    trj, self.r_min, min_bound=self.min_bound
                )
                if void_condition:
                    constrained_trajectories.append(trj)
                    distances.append(
                        self.env.get_particle_centroids(particles=particles)[
                            object_of_interest
                        ][0]
                    )
            # choose the sufficient trajectory that results in the min distance to centroid
            if distances:
                control_action = constrained_trajectories[np.argmin(distances)]

        if control_action is None:
            print(f"Error: No path satisfies void constraints. Choosing random path.")
            control_action = trajectories[np.random.randint(len(default_controls))]

        return control_action
