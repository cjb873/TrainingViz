import importlib.metadata

__version__ = importlib.metadata.version("TrainingViz")

from . import callback
from . import trainer
from . import viz
