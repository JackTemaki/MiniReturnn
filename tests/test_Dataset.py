# Also see test_SprintDataset.py.

from __future__ import annotations

import sys
import tests.setup_test_env  # noqa
import unittest
import numpy
from returnn.datasets.generating import Task12AXDataset
from returnn.datasets.basic import Dataset

import better_exchook


def test_Task12AXDataset_deepcopy():
    from copy import deepcopy

    dataset = Task12AXDataset(num_seqs=10)
    dataset = deepcopy(dataset)
    dataset.init_seq_order(1)
    n = dataset.num_seqs
    for i in range(n):
        dataset.load_seqs(i, i + 1)
        targets = dataset.get_data(i, "classes")
        print(targets)
    assert not dataset.is_less_than_num_seqs(n)


def test_Task12AXDataset_inf():
    dataset = Task12AXDataset(num_seqs=float("inf"))
    dataset.init_seq_order(1)
    n = 10
    for i in range(n):
        dataset.load_seqs(i, i + 1)
        targets = dataset.get_data(i, "classes")
        print(targets)
    assert dataset.is_less_than_num_seqs(n)


def test_Task12AXDataset_random():
    dataset = Task12AXDataset(num_seqs=10, seq_ordering="random")
    dataset.init_seq_order(1)
    n = dataset.num_seqs
    for i in range(n):
        dataset.load_seqs(i, i + 1)
        targets = dataset.get_data(i, "classes")
        print(targets)
    assert not dataset.is_less_than_num_seqs(n)


def test_get_seq_order():
    dataset = Dataset()
    num_seqs = 30

    def get_seq_len(i):
        return i**2 % 17  # some dummy lengths

    for seq_ordering in [
        "default",
        "default_every_n:5",
        "sorted",
        "sorted_reverse",
        "random:3",
        "laplace:3",
        "laplace:.10",
        "sort_bin_shuffle:3",
        "sort_bin_shuffle_x2:.10",
    ]:

        dataset.seq_ordering = seq_ordering

        # test full epoch
        dataset.partition_epoch = 1
        epoch = 3
        seq_index = dataset.get_seq_order_for_epoch(epoch, num_seqs, get_seq_len)

        assert isinstance(seq_index, (list, range, numpy.ndarray))
        assert len(set(seq_index)) == num_seqs  # right number of sequences, no duplicates

        # test partitioned epoch
        partition_epoch = 4
        dataset.partition_epoch = partition_epoch
        all_partitions_seq_index = []
        for epoch in range(1, partition_epoch + 1):
            partition_seq_index = dataset.get_seq_order_for_epoch(epoch, num_seqs, get_seq_len)
            all_partitions_seq_index += list(partition_seq_index)

        # Make sure partitions combined result in full epoch. This tests the random seed of Dataset which should be
        # fixed across partitions.
        assert set(all_partitions_seq_index) == set(seq_index)


if __name__ == "__main__":
    better_exchook.install()
    if len(sys.argv) <= 1:
        for k, v in sorted(globals().items()):
            if k.startswith("test_"):
                print("-" * 40)
                print("Executing: %s" % k)
                try:
                    v()
                except unittest.SkipTest as exc:
                    print("SkipTest:", exc)
                print("-" * 40)
        print("Finished all tests.")
    else:
        assert len(sys.argv) >= 2
        for arg in sys.argv[1:]:
            print("Executing: %s" % arg)
            if arg in globals():
                globals()[arg]()  # assume function and execute
            else:
                eval(arg)  # assume Python code and execute
