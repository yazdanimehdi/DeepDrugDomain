import threading
from queue import Queue
from typing import Iterable, Iterator
from torch.utils.data.sampler import Sampler


class PrefetchSampler(Sampler[int]):
    """
    Sampler that prefetches batch indices in the background.

    This sampler fetches batch indices in a background thread to minimize
    the data loading latency. Designed to be used with torch.utils.data.DataLoader.

    Attributes:
    - sampler (Sampler): An instance of a PyTorch sampler to prefetch from.
    - num_prefetch (int): Number of batches to prefetch in advance.
    - queue (Queue): A queue to store the prefetched batches.
    - stop_event (threading.Event): An event to signal the producer thread to stop.
    - thread (threading.Thread): The producer thread that prefetches batches in the background.
    """

    def __init__(self, sampler: Sampler[int], num_prefetch: int = 1):
        """
        Initializes the PrefetchSampler.

        Parameters:
        - sampler (Sampler): The base sampler to draw samples from.
        - num_prefetch (int, optional): Number of batches to prefetch in advance. Default: 1.
        """
        super().__init__(sampler.data_source)
        self.sampler = sampler
        self.num_prefetch = num_prefetch
        self.queue = Queue(maxsize=num_prefetch)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._produce)
        self.thread.start()

    def _produce(self) -> None:
        """
        Producer function that runs in a background thread,
        prefetching batches from the underlying sampler.
        """
        for batch in self.sampler:
            if self.stop_event.is_set():
                break
            self.queue.put(batch)

    def __iter__(self) -> Iterator[int]:
        """
        Yield batches of indices.

        Returns:
        - Iterator[int]: An iterator that yields batches of indices.
        """
        while not (self.stop_event.is_set() and self.queue.empty()):
            yield self.queue.get()

    def __len__(self) -> int:
        """
        Returns the number of elements in the base sampler.
        """
        return len(self.sampler)

    def stop(self) -> None:
        """Signal the producer thread to stop and clear the queue."""
        self.stop_event.set()
        while not self.queue.empty():
            self.queue.get()