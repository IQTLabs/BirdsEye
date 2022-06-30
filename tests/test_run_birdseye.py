import sys

from run_birdseye import main


def test_run_birdseye_main():
    sys.argv = ['', '-c', 'tests/test_mcts_multi.yaml']
    main()
