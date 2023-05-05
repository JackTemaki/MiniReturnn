import tests.setup_test_env  # noqa
from returnn.config import Config
import returnn.util.basic as util
from returnn.datasets.sprint import ExternSprintDataset
import os
import sys
import unittest
import better_exchook

dummyconfig_dict = {
    "num_inputs": 2,
    "num_outputs": 3,
    "hidden_size": (1,),
    "hidden_type": "forward",
    "activation": "relu",
    "bidirectional": False,
}


os.chdir((os.path.dirname(__file__) or ".") + "/..")
assert os.path.exists("rnn.py")
sprintExecPath = "tests/DummySprintExec.py"


def test_read_all():
    config = Config()
    config.update(dummyconfig_dict)
    print("Create ExternSprintDataset")
    python_exec = util.which("python")
    if python_exec is None:
        raise unittest.SkipTest("python not found")
    num_seqs = 4
    dataset = ExternSprintDataset(
        [python_exec, sprintExecPath],
        "--*.feature-dimension=2 --*.trainer-output-dimension=3 "
        "--*.crnn-dataset=DummyDataset(2,3,num_seqs=%i,seq_len=10)" % num_seqs,
    )
    dataset.init_seq_order(epoch=1)
    seq_idx = 0
    while dataset.is_less_than_num_seqs(seq_idx):
        dataset.load_seqs(seq_idx, seq_idx + 1)
        for key in dataset.get_data_keys():
            value = dataset.get_data(seq_idx, key)
            print("seq idx %i, data %r: %r" % (seq_idx, key, value))
        seq_idx += 1
    assert seq_idx == num_seqs


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
