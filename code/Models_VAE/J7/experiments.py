from abc import ABC, abstractmethod
from output import Log

"""
Base class to describe an experiment.
"""
class Experiment(ABC):

    def __init__(self, cfg):
        self.cfg = cfg
        self.log = Log(self.cfg.LOG_DIR)
        self.log.log_config(cfg)

    @abstractmethod
    def train(self):
        """Train a new instance of the model."""
        raise NotImplementedError()

    @abstractmethod
    def eval(self, load_model_name):
        """Evaluate a saved instance of the model."""
        raise NotImplementedError()


