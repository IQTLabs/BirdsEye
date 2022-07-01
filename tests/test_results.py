"""
Tests for results.py
"""
from birdseye.results import plotter
from birdseye.results import separate_plotter
from birdseye.results import single_metric_grid
from birdseye.results import single_plot
from birdseye.results import std_dev_grid
from birdseye.results import two_metric_grid


def test_plotter():
    plotter(two_metric_grid, metric1='r_err', metric2='theta_err', limit=2, variance_bars=False)
    plotter(single_std_dev, metric1='r_err', metric2='theta_err', limit=2, variance_bars=False)
    plotter(std_dev_grid, metric1='r_err', metric2='theta_err', limit=2, variance_bars=False)


def test_separate_plotter():
    separate_plotter(single_metric_grid ,metric1='r_err', metric2='theta_err', timing=False, variance_bars=True, limit=2)
