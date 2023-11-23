import birdseye.utils
import birdseye.sensor
import birdseye.actions
import birdseye.state
import birdseye.env
import birdseye.mcts_utils


class LightMCTS:
    def __init__(self, env, depth=3, c=20, simulations=100, n_downsample=400):
        self.env = env
        self.depth = depth
        self.c = c
        self.simulations = simulations
        self.n_downsample = n_downsample

        self.Q = {}
        self.N = {}

        self.action = None

    def get_action(
        self,
    ):
        self.Q, self.N, self.action = birdseye.mcts_utils.select_action_light(
            self.env,
            self.Q,
            self.N,
            self.depth,
            self.c,
            self.simulations,
            self.n_downsample,
        )
        birdseye.mcts_utils.trim_tree(
            self.Q, self.N, self.env.actions.action_to_index(self.action)
        )

        return [list(self.action)]
