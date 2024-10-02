from aviary.env import TASK_DATASET_REGISTRY
from aviary.env import DummyTaskDataset as _DummyTaskDataset

from .callbacks import ComputeTrajectoryMetricsMixin


class DummyTaskDataset(_DummyTaskDataset, ComputeTrajectoryMetricsMixin):
    pass


TASK_DATASET_REGISTRY["dummy"] = "ldp.alg.datasets", "DummyTaskDataset"
