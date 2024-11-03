import random
from collections import UserList


class CircularReplayBuffer(UserList[dict]):
    def resize(self, size: int):
        if len(self) > size:
            self.data = self.data[-size:]

    def batched_iter(
        self, batch_size: int, shuffle: bool = True, infinite: bool = False
    ):
        while True:
            indices = list(range(len(self)))
            if shuffle:
                random.shuffle(indices)

            for i in range(0, len(self), batch_size):
                keys = self.data[0].keys()

                batch: dict[str, list] = {k: [] for k in keys}
                for j in indices[i : i + batch_size]:
                    if self.data[j].keys() != keys:
                        raise RuntimeError(
                            "Found buffer element with inconsistent keys"
                        )

                    for k in keys:
                        batch[k].append(self.data[j][k])

                yield batch

            if not infinite:
                break


class RandomizedReplayBuffer(CircularReplayBuffer):
    def resize(self, size: int):
        if len(self) > size:
            self.data = random.sample(self.data, size)
