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
