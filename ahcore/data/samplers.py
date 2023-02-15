# encoding: utf-8
"""Module implementing the samplers. These are used for instance to create batches of the same WSI.
"""
from __future__ import annotations

import math

import torch
import torch.utils.data
from dlup.data.dataset import ConcatDataset

from ahcore.utils.io import get_logger

logger = get_logger()


class WsiBatchSampler(torch.utils.data.Sampler[list[int]]):
    def __init__(self, dataset: ConcatDataset, sampler: torch.utils.data.Sampler, batch_size: int):
        super().__init__(data_source=dataset)
        self._dataset = dataset
        self._sampler = sampler
        self._batch_size = batch_size
        self._num_batches = 0

        self._slices: list[slice] = []
        self._compute_slices()
        self._slice_iterator = iter(self._slices[1:])
        self.__next_value = self._slices[0].stop

    def _compute_slices(self):
        for idx, dataset in enumerate(self._dataset.datasets):
            slice_start = 0 if len(self._slices) == 0 else self._slices[-1].stop
            slice_stop = self._dataset.cumulative_sizes[idx]
            self._slices.append(slice(slice_start, slice_stop))
            self._num_batches += math.ceil(len(dataset) / self._batch_size)

    def __iter__(self):
        batch = []
        for idx in self._sampler:
            batch.append(idx)
            if (len(batch) == self._batch_size) or (idx == self.__next_value - 1):
                yield batch
                batch = []

            if idx == self.__next_value - 1:
                try:
                    self.__next_value = next(self._slice_iterator).stop
                except StopIteration:
                    pass

        if len(batch) > 0:
            yield batch

    def __len__(self):
        return self._num_batches

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"batch_size={self._batch_size}, "
            f"num_batches={self._num_batches}, "
            f"num_wsis={len(self._dataset.datasets)})"
        )
