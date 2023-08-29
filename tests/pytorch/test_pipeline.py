"""
Tests for PyTorch data pipeline.
"""

import tests.setup_test_env  # noqa
import torch

from returnn.config import Config
from returnn.datasets import init_dataset
from returnn.torch.engine import Engine


def test_min_seq_len():

    from returnn.datasets.generating import DummyDataset

    config = Config({"min_seq_length": 2})
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=1, seq_len=1)
    engine = Engine(config=config)
    data_loader = engine._create_data_loader(dataset)
    for _ in data_loader:
        assert False, "Should not contain sequences"

    config = Config()
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=1, seq_len=3)
    engine = Engine(config=config)
    data_loader = engine._create_data_loader(dataset)
    for _ in data_loader:
        return
    assert False, "Should have contained sequences"


def test_max_seq_len():

    from returnn.datasets.generating import DummyDataset

    config = Config({"max_seq_length": 4})
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=1, seq_len=5)
    engine = Engine(config=config)
    data_loader = engine._create_data_loader(dataset)
    for _ in data_loader:
        assert False, "Should not contain sequences"

    config = Config()
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=1, seq_len=3)
    engine = Engine(config=config)
    data_loader = engine._create_data_loader(dataset)
    for _ in data_loader:
        return
    assert False, "Should have contained sequences"


def test_multiple_workers():

    from returnn.datasets.generating import DummyDataset

    config = Config({"max_seqs": 8, "num_workers_per_gpu": 1, "batch_size": 8000})
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=8, seq_len=5)
    engine = Engine(config=config)
    data_loader = engine._create_data_loader(dataset)
    # we had 8 sequences and a max batch size of 8, one worker, so one batch of 8
    i = 0
    for i, batch in enumerate(data_loader):
        assert len(batch["data"]) == 8
    assert i == 0  # should have had one batch

    config = Config({"max_seqs": 8, "num_workers_per_gpu": 2, "batch_size": 8000})
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=8, seq_len=5)
    engine = Engine(config=config)
    data_loader = engine._create_data_loader(dataset)
    # we had 8 sequences and a max batch size of 8, but as we have two workers
    # we should get two batches of 4
    i = 0
    for i, batch in enumerate(data_loader):
        assert len(batch["data"]) == 4
        # test if sequence tags are interleaved as (0,2,4,6) and (1,3,5,7)
        if i == 0:
            assert batch["seq_tag"] == ["seq-0", "seq-2", "seq-4", "seq-6"]
        if i == 1:
            assert batch["seq_tag"] == ["seq-1", "seq-3", "seq-5", "seq-7"]

    assert i == 1  # should have had two batches


