from typing import *
from time import perf_counter
import os
import os.path as osp

# from torch.utils.data import IterableDataset
from mindspore.dataset import Dataset

from datasets import load_dataset, DatasetDict
from contextlib import contextmanager

import ray
from ray.actor import ActorHandle
from asyncio import Event

from tqdm import tqdm


def list_gz_in_dir(data_dir):
    result = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.gz'):
                result.append(osp.join(root, file))
    return result


def list_dir_with(data_dir, filter_func: Callable[[str], bool]):
    result = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if filter_func(file):
                result.append(osp.join(root, file))
    return result


def load_pytorrent(root_dir):
    data_files = {
        split_name: list_gz_in_dir(osp.join(root_dir, split_name))
        for split_name in ['train', 'valid', 'test']
    }

    return load_dataset('json', data_files=data_files)


def load_codeparrot(root_dir):
    def is_data_file(name):
        return len(name) == 3 and all(x.isdigit() for x in name)

    data_files = {
        'train': list_gz_in_dir(root_dir),
    }

    return load_dataset('json', data_files=data_files, streaming=True)


def load_apps(root_dir):
    data_files = {
        split_name: osp.join(root_dir, split_name + '.jsonl')
        for split_name in ['train', 'test']
    }
    return load_dataset('json', data_files=data_files, streaming=True)

@contextmanager
def time_it(desc: str):
    tic = perf_counter()
    try:
        print(desc)
        yield 0
    finally:
        toc = perf_counter()
        print(f'done, took {toc - tic:.4f} sec')


class MyIterableDataset(Dataset):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset

    def __iter__(self):
        return iter(self.raw_dataset)


def iter_wrapper(ds: Iterable):
   return MyIterableDataset(ds)


@ray.remote
class ProgressActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return
