import os

# Local location of BirdsEye src directory
BIRDSEYE_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory containing entire BirdsEye repo
REPO_DIR = os.path.split(BIRDSEYE_SRC_DIR)[0]

# Local directory containing run information
RUN_DIR = os.path.join(REPO_DIR, 'runs')
