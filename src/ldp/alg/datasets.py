from aviary.core import TASK_DATASET_REGISTRY
from aviary.core import DummyTaskDataset as _DummyTaskDataset

from .callbacks import ComputeTrajectoryMetricsMixin


class DummyTaskDataset(_DummyTaskDataset, ComputeTrajectoryMetricsMixin):
    pass


TASK_DATASET_REGISTRY["dummy"] = "ldp.alg.datasets", "DummyTaskDataset"
