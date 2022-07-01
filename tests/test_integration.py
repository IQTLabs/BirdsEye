"""
Integration tests
"""
import os
import tempfile
import unittest

from run_birdseye import run_birdseye


TEST_CONFIG = """
[Methods]
env : RFMultiEnv
method : baseline
action : baselineactions
sensor : doublerssilofi
state : rfmultistate
target_speed: 1.0
target_start: 100
fading_sigma: 8.0
particle_resample: 0.01
particle_min: 10
particle_max: 200
delta_col: 15
likelihood: lofiv2

[Defaults]
policy : static
plotting : False
trials : 2
timesteps : 2
"""


class FakeArgs:
    """
    Mock class to provide fake args
    """

    def __init__(self, config):
        """
        Initialize variables
        """
        self.config = config


class BirdsEyeIntegrationTest(unittest.TestCase):
    """
    Class of integration tests of running birdseye
    """

    def test_integration(self):
        """
        Integration test
        """
        with tempfile.TemporaryDirectory() as tempdir:
            config_file = os.path.join(tempdir, 'config.yaml')
            with open(config_file, 'w', encoding='UTF-8') as file:
                file.write(TEST_CONFIG)
            fake_args = FakeArgs(config_file)
            run_birdseye(args=fake_args)
