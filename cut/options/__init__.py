"""This package options includes option modules: training options, test options, and basic options (used in both training and test)."""

from cut.options.train_options import TrainOptions
from cut.options.test_options import TestOptions

__all__ = ['TrainOptions', 'TestOptions']
