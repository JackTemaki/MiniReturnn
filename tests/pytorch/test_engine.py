from returnn.config import Config
from returnn.torch.engine import Engine
import torch
from torch import nn
import tempfile


def ce_train_step(*, model, data, run_ctx, **_kwargs):
    features = data["data"]
    features_len = data["data:size1"]
    features_len, indices = torch.sort(features_len, descending=True)
    features = features[indices, :, :]

    labels = data["classes"][indices, :]
    labels_len = data["classes:size1"][indices]

    logits = model(features)

    loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
    num_frames = torch.sum(labels_len)
    run_ctx.mark_as_loss(name="CE", loss=loss, inv_norm_factor=num_frames)


class LinearModel(nn.Module):
    """
    A minimal test module
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, x):
        x = x.type(torch.float32)
        logits = self.linear(x)
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        return logits_ce_order


def get_linear_model(epoch, step, **kwargs):
    return LinearModel()


def test_engine_train():
    from returnn.datasets.generating import DummyDataset

    seq_len = 5
    n_data_dim = 2
    n_classes_dim = 3
    train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
    train_data.init_seq_order(epoch=1)
    cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
    cv_data.init_seq_order(epoch=1)

    with tempfile.TemporaryDirectory() as d:
        config = Config()
        config.update(
            {
                "model": "%s/model" % d,
                "num_epochs": 5,
                "optimizer": {"class": "adam"},
                "get_model": get_linear_model,
                "train_step": ce_train_step,
            }
        )

        engine = Engine(config)
        engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data)
        engine.train()
