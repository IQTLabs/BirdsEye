"""
Tests for run_birdseye.py
"""
import sys

from run_birdseye import main


def test_run_birdseye_main():
    """
    Test the main function
    """
    sys.argv = ["", "-c", "tests/test_mcts_multi.yaml"]
    main()
