from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("chicell2cell")
except PackageNotFoundError:
    __version__ = "0.1.0"

from . import preprocessing, graph, model, communication, visualization, comparison

__all__ = ["preprocessing", "graph", "model", "communication", "visualization", "comparison"]
