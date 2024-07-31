# Also see test_SprintDataset.py.

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
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


def test_get_seq_order_laplace_reference():
    num_seqs = 3023
    rnd = numpy.random.RandomState(42)
    seq_lens = rnd.randint(1, 23, size=[num_seqs])
    get_seq_len = seq_lens.__getitem__

    dataset = Dataset()
    dataset.epoch = 1
    dataset.seq_ordering = "laplace:.100"

    seq_index_ = dataset.get_seq_order_for_epoch(epoch=1, num_seqs=num_seqs, get_seq_len=get_seq_len)
    assert isinstance(seq_index_, (list, range, numpy.ndarray))
    assert len(set(seq_index_)) == num_seqs  # right number of sequences, no duplicates
    print("current implementation returns seq_lens[seq_index]:", list(seq_lens[seq_index_]))

    tmp = dataset.seq_ordering.split(":")[1:]
    if len(tmp) == 0:
        bins = 2
    else:
        if tmp[0].startswith("."):  # starting with "." -> approx chunk size (num of seqs in one bin)
            bins = max(num_seqs // int(tmp[0][1:]), 2)
        else:  # the number of bins
            bins = int(tmp[0])
    assert len(tmp) <= 1
    rnd_seed = dataset.epoch
    random_generator = numpy.random.RandomState(rnd_seed)
    seq_index = random_generator.permutation(num_seqs)
    out_index = []
    for i in range(bins):
        if i == bins - 1:
            part = seq_index[i * len(seq_index) // bins :].tolist()
        else:
            part = seq_index[i * len(seq_index) // bins : (i + 1) * len(seq_index) // bins].tolist()
        part.sort(key=get_seq_len, reverse=(i % 2 == 1))
        out_index += part
    seq_index = out_index
    print("reference seq_lens[seq_index]:", list(seq_lens[seq_index]))

    assert len(seq_index) == num_seqs == len(seq_index_)
    assert seq_index == list(seq_index_)


@contextlib.contextmanager
def create_ogg_zip_txt_only_dataset(*, text: str = "hello world", seq_tag: str = "sequence0.wav"):
    import zipfile
    from returnn.datasets.audio import OggZipDataset

    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp_zip_file, tempfile.NamedTemporaryFile(
        suffix=".txt"
    ) as tmp_vocab_file:
        with zipfile.ZipFile(tmp_zip_file.name, "w") as zip_file:
            zip_file.writestr(
                os.path.basename(tmp_zip_file.name)[:-4] + ".txt",
                repr([{"text": text, "duration": 2.3, "file": seq_tag}]),
            )
        vocab = {"@": 2, " ": 1, ".": 0}
        vocab.update({chr(i): i - ord("a") + 3 for i in range(ord("a"), ord("z") + 1)})
        tmp_vocab_file.write(repr(vocab).encode("utf8"))
        tmp_vocab_file.flush()

        dataset = OggZipDataset(
            path=tmp_zip_file.name,
            audio=None,
            targets={"class": "CharacterTargets", "vocab_file": tmp_vocab_file.name, "seq_postfix": [0]},
        )
        dataset.initialize()
        yield dataset


def test_OggZipDataset():
    from returnn.datasets.audio import OggZipDataset

    _demo_txt = "some utterance text"

    with create_ogg_zip_txt_only_dataset(text=_demo_txt) as dataset:
        assert isinstance(dataset, OggZipDataset)
        dataset.init_seq_order(epoch=1)
        dataset.load_seqs(0, 1)
        raw = dataset.get_data(0, "raw")
        orth = dataset.get_data(0, "orth")
        classes = dataset.get_data(0, "classes")
        print("raw:", raw)
        print("orth:", orth)
        print("classes:", classes)
        assert isinstance(raw, numpy.ndarray) and raw.dtype.name.startswith("str") and raw.shape == ()
        raw_ = raw.item()
        assert isinstance(raw_, str) and raw_ == _demo_txt
        assert isinstance(orth, numpy.ndarray) and orth.dtype == numpy.uint8 and orth.ndim == 1
        orth_ = orth.tostring()
        assert orth_.decode("utf8") == _demo_txt
        assert isinstance(classes, numpy.ndarray) and classes.dtype == numpy.int32 and classes.ndim == 1
        classes_ = "".join([dataset.targets.id_to_label(c) for c in classes])
        assert classes_ == _demo_txt + "."


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
