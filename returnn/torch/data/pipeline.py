"""
Code to create PyTorch datasets that can be used with the PyTorch DataLoader.

We make use of TorchData data pipelines.

Most functionality is implemented as a dataset/datapipe, as this seems to be the common way in PyTorch,
as it is also commonly done in Fairseq:
    https://github.com/facebookresearch/fairseq/tree/main/fairseq/data
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/subsample_dataset.py

This is also the intended way for TorchData.

We potentially could also implement some functionality as part of the data loader (v1),
but DataLoader2 suggests to decouple this, as we do here.

We also have :class:`ChunkShuffleDataset` on RETURNN dataset level.
However, having this separate pure PyTorch implementation is useful to allow to use
other PyTorch datasets more directly, including also HuggingFace datasets.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Union
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.utils.data

from returnn.util.basic import NumbersDict

InputType = Union[np.ndarray, int, str, float, bool]
OutputType = Union[torch.Tensor, int, str, float, bool]


def create_tensor(value: InputType) -> OutputType:
    """
    Only returnn np.ndarray values as tensor, and adjust non-supported dtypes

    Other formats, such as "int" (e.g. seq_idx) or "str" (e.g. seq_tag) are returned as is.

    :param value: e.g. np.ndarray to be converted
    """
    if not isinstance(value, np.ndarray):
        return value

    # The only supported PyTorch dtypes are:
    # float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
    if value.dtype.kind in "UO":  # string (unicode) or object in numpy array
        return value  # keep as-is
    if value.dtype == np.uint32:
        value = np.asarray(value, dtype=np.int64)
    elif value.dtype == np.uint16:
        value = np.asarray(value, dtype=np.int32)
    return torch.tensor(value)


def collate_batch(batch: List[Dict[str, InputType]], device: str = "cpu") -> Dict[str, OutputType]:
    """
    Use with `functools.partial` to set the device!

    :param batch: the batch as list to collate into single Tensors
    :param device: the target device to move the Tensor to
    """
    assert isinstance(batch, list)
    assert batch, "batch is empty?"
    assert isinstance(batch[0], dict)
    data_keys = list(batch[0].keys())

    res = {}
    for key in data_keys:
        ls = [create_tensor(sample[key]) for sample in batch]
        if not isinstance(ls[0], torch.Tensor):
            # no padding for non-Tensor types
            res[key] = ls
            continue
        num_axis = len(ls[0].size())
        if num_axis > 0:
            padded = torch.nn.utils.rnn.pad_sequence(ls, batch_first=True, padding_value=0)
            for i in range(num_axis):
                res["%s:size%i" % (key, i + 1)] = torch.tensor([v.shape[i] for v in ls])
        else:
            padded = torch.stack(ls)
        res[key] = padded
        res["%s:size0" % key] = torch.tensor(len(ls))

    return res


class ChunkingIterDataPipe(torch.utils.data.IterDataPipe):
    """
    Splits each sequence in the given dataset into chunks according to the 'chunking' config option.
    So it transforms one sequences into multiple sequences.
    """

    def __init__(self, dataset: torch.utils.data.IterableDataset, chunking_options):
        """
        :param dataset: dataset to apply chunking to
        :param chunking_options: dictionary in the following format
            {
                chunk_streams: {
                    "data": {
                        "size": ...,
                        "step": ...,
                        "min_chunk_size": ...,
                    },
                    "classes": {
                        "size": ...,
                        "step": ...,
                        "min_chunk_size": ...,
                    }
                }
                "random_chunk_start": True/False  # Start within [0, chunk_step] for first chunk
            }
        """
        super().__init__()
        self._dataset = dataset
        assert "chunk_streams" in chunking_options
        assert len(chunking_options["chunk_streams"]) > 0
        self._chunking_data_keys = list(chunking_options["chunk_streams"].keys())
        self._chunk_size = NumbersDict({key: entry["size"] for key, entry in chunking_options["chunk_streams"].items()})
        self._chunk_step = NumbersDict({key: entry["step"] for key, entry in chunking_options["chunk_streams"].items()})
        self._min_chunk_size = NumbersDict(
            {key: entry.get("min_chunk_size") for key, entry in chunking_options["chunk_streams"].items()}
        )
        self._random_chunk_start = chunking_options.get("random_chunk_start", False)

    def __iter__(self) -> Iterable[List[Dict[str, InputType]]]:
        """
        :return: generator providing chunks in the form of a dict data_key -> data chunk
        """
        for data_dict in self._dataset:
            data_chunks = {}
            num_chunks = None  # to verify number of chunks
            if self._random_chunk_start:
                start = np.random.random()
            else:
                start = 0

            for data_key in self._chunking_data_keys:
                chunk_size = self._chunk_size[data_key]
                chunk_step = self._chunk_step[data_key]
                min_chunk_size = self._min_chunk_size[data_key]

                data = data_dict[data_key]
                chunks = [
                    data[start_index : start_index + chunk_size]
                    for start_index in range(int(start * chunk_step), len(data), chunk_step)
                    if len(data[start_index : start_index + chunk_size]) >= min_chunk_size
                ]

                if num_chunks is None:
                    num_chunks = len(chunks)
                else:
                    assert num_chunks == len(
                        chunks
                    ), "Chunking resulted in different number of chunks for different data keys."

                data_chunks[data_key] = chunks

            if num_chunks == 0:
                continue
            for chunk_index in range(num_chunks):
                chunk_data = {data_key: data_chunks[data_key][chunk_index] for data_key in data_chunks.keys()}

                # If chunking is configured using a dict,
                # i.e. with explicit data keys, there might be remaining data keys
                # for which we yield the full sequence in each chunk.
                non_chunked_data = {
                    data_key: data for data_key, data in data_dict.items() if data_key not in chunk_data
                }
                if non_chunked_data:
                    chunk_data.update(deepcopy(non_chunked_data))

                yield chunk_data

    def __getitem__(self, index):
        raise Exception(f"{self.__class__.__name__}.__getitem__ not supported")


# noinspection PyAbstractClass
class BatchingIterDataPipe(torch.utils.data.IterDataPipe):
    """
    Converts a dataset yielding sequences (dict data_key -> array per sequence) into a dataset yielding lists of
    these sequences, i.e. batches.
    Sequences are grouped in-order according to the 'max_tokens' and 'max_seqs' batch size
    limits.
    Note, that batches are not yet merged into a single (padded) data array here, this happens in 'collate_batch()'.
    """

    def __init__(self, dataset: torch.utils.data.IterableDataset, batch_size=1, max_seqs=None, drop_last=False):
        """
        :param dataset: dataset to apply batching to
        :param int|dict[str,int]|None batch_size: Maximum number of time steps (e.g. audio frames / words) in one
            batch (padding included).
            If given as a dict data_key -> value, sets different individual limits per data key.
            If None, no limit.
        :param int|None max_seqs: maximum number of sequences in a batch,
            None means unlimited (also -1 to match TF backend)
        :param drop_last: if true, drop the last (possibly incomplete) batch.
        """
        super().__init__()
        self._dataset = dataset
        self._max_batch_size = NumbersDict(sys.maxsize if batch_size is None else batch_size)
        self._max_seqs = sys.maxsize if (max_seqs is None or max_seqs == -1) else max_seqs
        self._drop_last = drop_last

        assert self._max_batch_size.min_value() > 0
        assert self._max_seqs > 0

    def __iter__(self) -> Iterable[List[Dict[str, InputType]]]:
        """
        :return: generator providing batches in the form of lists of sequences, where each sequence is a dict
          data_key -> data_array.
        """
        current_batch = []
        current_max_sequence_lengths = NumbersDict(0)  # data_key -> length of longest sequence in current batch

        for data_dict in self._dataset:
            if len(current_batch) == self._max_seqs:
                yield current_batch
                current_batch = []
                current_max_sequence_lengths = NumbersDict(0)

            # TODO: This assumes all data has time as first dimension. Currently we can't know better..
            # Scalars are treated as length 1
            sequence_lengths = NumbersDict(
                {
                    data_key: (data.shape[0] if isinstance(data, np.ndarray) and len(data.shape) > 0 else 1)
                    for data_key, data in data_dict.items()
                }
            )

            max_sequence_lengths_if_included = NumbersDict.max([current_max_sequence_lengths, sequence_lengths])
            batch_size_if_included = max_sequence_lengths_if_included * (len(current_batch) + 1)  # including padding

            if current_batch and batch_size_if_included.any_compare(self._max_batch_size, (lambda a, b: a > b)):
                yield current_batch
                current_batch = [data_dict]
                current_max_sequence_lengths = sequence_lengths
            else:
                current_batch.append(data_dict)
                current_max_sequence_lengths = max_sequence_lengths_if_included

        if current_batch and not self._drop_last:
            yield current_batch


class LenFilterDataPipe(torch.utils.data.IterDataPipe):
    """
    Removes sequences which are either too long or too short from a dataset
    Returns dataset yielding list of data lengths within the defined range
    """

    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        min_seq_length: Union[int, NumbersDict] = None,
        max_seq_length: Union[int, NumbersDict] = None,
    ):
        """
        :param dataset: dataset to apply the filter to
        :param min_seq_length: minimum sequence length either in general or per data_key via dict
        :param max_seq_length: maximum sequence length either in general or per data_key via dict
        """
        super().__init__()
        self._dataset = dataset
        self._min_seq_length = NumbersDict(0 if min_seq_length is None else min_seq_length)
        self._max_seq_length = NumbersDict(sys.maxsize if max_seq_length is None else max_seq_length)

    def __iter__(self):
        """
        :return: generator providing filtered data where each sequence is a dict
          data_key -> data_array.
        :rtype: Iterable[dict[str, numpy.ndarray]]
        """
        for data_dict in self._dataset:
            # TODO: This assumes all data has time as first dimension. Currently we can't know better..
            sequence_lengths = NumbersDict(
                {
                    data_key: data.shape[0]
                    for data_key, data in data_dict.items()
                    if isinstance(data, np.ndarray) and data.shape
                }
            )
            if sequence_lengths.any_compare(self._min_seq_length, lambda a, b: a < b):
                continue
            if sequence_lengths.any_compare(self._max_seq_length, lambda a, b: a > b):
                continue
            yield data_dict

    def __getitem__(self, index):
        raise Exception(f"{self.__class__.__name__}.__getitem__ not supported")
