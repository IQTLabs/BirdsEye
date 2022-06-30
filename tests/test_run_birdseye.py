"""
Tests for run_birdseye.py
"""
import sys

from run_birdseye import main


def test_run_birdseye_main():
    """
    Test the main function
    """
    sys.argv = ['', '-c', 'tests/test_mcts_multi.yaml']
    main()


def test_run_birdseye_main_batch():
    """
    Test the main function using batch mode
    """
    sys.argv = ['', '-c', 'tests/test_mcts_multi.yaml', '-b']
    main()
