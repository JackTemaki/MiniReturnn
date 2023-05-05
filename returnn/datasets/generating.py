"""
Some datasets for artificially generated data.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy
import sys
import typing

from returnn.util.basic import class_idx_seq_to_1_of_k, CollectionReadCheckCovered
from returnn.log import log

from .util.feature_extraction import ExtractAudioFeatures
from .util.vocabulary import *
from .audio import OggZipDataset  # noqa # for API compatibility
from .basic import Dataset, DatasetSeq, convert_data_dims
from .cached2 import CachedDataset2


class GeneratingDataset(Dataset):
    """
    Some base class for datasets with artificially generated data.
    """

    _input_classes = None
    _output_classes = None

    def __init__(self, input_dim, output_dim, num_seqs=float("inf"), **kwargs):
        """
        :param int|None input_dim:
        :param int|dict[str,int|(int,int)|dict] output_dim: if dict, can specify all data-keys
        :param int|float num_seqs:
        """
        super(GeneratingDataset, self).__init__(**kwargs)
        assert self.shuffle_frames_of_nseqs == 0

        self._input_dim = input_dim
        self._output_dim = output_dim
        self.num_inputs = input_dim
        output_dim = convert_data_dims(output_dim, leave_dict_as_is=False)
        if "data" not in output_dim and input_dim is not None:
            output_dim["data"] = (input_dim, 2)  # not sparse
        self.num_outputs = output_dim
        self.expected_load_seq_start = 0
        self._seq_order = None  # type: Optional[Sequence[int]]
        self._num_seqs = num_seqs
        self._total_num_seqs = num_seqs
        self.random = numpy.random.RandomState(1)
        self.reached_final_seq = False
        self.added_data = []  # type: typing.List[DatasetSeq]
        if self.seq_ordering in ("sorted", "sorted_reverse"):
            # For the dev/eval dataset, RETURNN automatically tries to sort them.
            # As this is not supported, just ignore it and reset it to the default order.
            self.seq_ordering = "default"

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """
        :type epoch: int|None
        :param list[str]|None seq_list: predefined order via tags, doesn't make sense here
        :param list[int]|None seq_order: predefined order via indices, doesn't make sense here
        This is called when we start a new epoch, or at initialization.
        """
        super(GeneratingDataset, self).init_seq_order(epoch=epoch)
        assert seq_list is None and seq_order is None, (
            "predefined order doesn't make sense for %s" % self.__class__.__name__
        )
        self.random.seed(self._get_random_seed_for_epoch(epoch=epoch))
        if self._total_num_seqs == float("inf"):
            assert self.seq_ordering == "default"
            self._seq_order = None
            self._num_seqs = self._total_num_seqs
        else:
            self._seq_order = self.get_seq_order_for_epoch(epoch=epoch, num_seqs=self._total_num_seqs, get_seq_len=None)
            self._num_seqs = len(self._seq_order)
        self._num_timesteps = 0
        self.reached_final_seq = False
        self.expected_load_seq_start = 0
        self.added_data = []
        return True

    def _cleanup_old_seqs(self, seq_idx_end):
        i = 0
        while i < len(self.added_data):
            if self.added_data[i].seq_idx >= seq_idx_end:
                break
            i += 1
        del self.added_data[:i]

    def _check_loaded_seq_idx(self, seq_idx):
        if not self.added_data:
            raise Exception("no data loaded yet")
        start_loaded_seq_idx = self.added_data[0].seq_idx
        end_loaded_seq_idx = self.added_data[-1].seq_idx
        if seq_idx < start_loaded_seq_idx or seq_idx > end_loaded_seq_idx:
            raise Exception(
                "seq_idx %i not in loaded seqs range [%i,%i]" % (seq_idx, start_loaded_seq_idx, end_loaded_seq_idx)
            )

    def _get_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq|None
        """
        for data in self.added_data:
            if data.seq_idx == seq_idx:
                return data
        return None

    def is_cached(self, start, end):
        """
        :param int start:
        :param int end:
        :rtype: bool
        """
        # Always False, to force that we call self._load_seqs().
        # This is important for our buffer management.
        return False

    def _load_seqs(self, start, end):
        """
        :param int start: inclusive seq idx start
        :param int end: exclusive seq idx end
        """
        # We expect that start increase monotonic on each call
        # for not-yet-loaded data.
        # This will already be called with _load_seqs_superset indices.
        assert start >= self.expected_load_seq_start
        if start > self.expected_load_seq_start:
            # Cleanup old data.
            self._cleanup_old_seqs(start)
            self.expected_load_seq_start = start
        if self.added_data:
            start = max(self.added_data[-1].seq_idx + 1, start)
        if end > self.num_seqs:
            end = self.num_seqs
        if end >= self.num_seqs:
            self.reached_final_seq = True
        seqs = [self._make_seq(seq_idx) for seq_idx in range(start, end)]
        self._num_timesteps += sum([seq.num_frames for seq in seqs])
        self.added_data += seqs

    def _make_seq(self, seq_idx: int) -> DatasetSeq:
        seq = self.get_corpus_seq(self.get_corpus_seq_idx(seq_idx))
        seq.seq_idx = seq_idx
        return seq

    def have_get_corpus_seq(self) -> bool:
        """
        :return: whether we have :func:`get_corpus_seq`
        """
        return True

    def get_corpus_seq(self, corpus_seq_idx: int) -> DatasetSeq:
        """
        :param corpus_seq_idx:
        :return: seq
        """
        # seed value based on epoch and corpus_seq_idx in order to get deterministic behavior
        self.random.seed((self._get_random_seed_for_epoch(epoch=self.epoch), corpus_seq_idx))
        seq = self.generate_seq(corpus_seq_idx)
        return seq

    def generate_seq(self, seq_idx: int) -> DatasetSeq:
        """
        This assumes that self.random is already initialized and seeded
        to sth deterministic for the given seq_idx and epoch.

        :param seq_idx: corpus seq idx
        """
        raise NotImplementedError

    def _shuffle_frames_in_seqs(self, start, end):
        assert False, "Shuffling in GeneratingDataset does not make sense."

    def get_num_timesteps(self):
        """
        :rtype: int
        """
        assert self.reached_final_seq
        return self._num_timesteps

    @property
    def num_seqs(self) -> int:
        """
        :return: num seqs for current epoch
        """
        return self._num_seqs

    def get_total_num_seqs(self) -> int:
        """
        :return: total num seqs
        """
        return self._total_num_seqs

    def have_corpus_seq_idx(self):
        """
        :return: whether we have :func:`get_corpus_seq_idx`
        """
        return True

    def get_corpus_seq_idx(self, seq_idx: int) -> int:
        """
        :param seq_idx:
        :return: corpus seq idx
        """
        if self._seq_order is None:
            return seq_idx
        return self._seq_order[seq_idx]

    def get_seq_length(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: returnn.util.NumbersDict
        """
        # get_seq_length() can be called before the seq is loaded via load_seqs().
        # Thus, we just call load_seqs() ourselves here.
        assert seq_idx >= self.expected_load_seq_start
        self.load_seqs(self.expected_load_seq_start, seq_idx + 1)
        return self._get_seq(seq_idx).num_frames

    def get_data(self, seq_idx, key):
        """
        :param int seq_idx:
        :param str key:
        :rtype: numpy.ndarray
        """
        return self._get_seq(seq_idx).features[key]

    def get_input_data(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: numpy.ndarray
        """
        return self.get_data(seq_idx, "data")

    def get_targets(self, target, seq_idx):
        """
        :param int seq_idx:
        :param str target:
        :rtype: numpy.ndarray
        """
        return self.get_data(seq_idx, target)

    def get_tag(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: str
        """
        self._check_loaded_seq_idx(seq_idx)
        return self._get_seq(seq_idx).seq_tag

    def get_all_tags(self):
        """
        :rtype: list[str]
        """
        return ["seq-%i" % seq_idx for seq_idx in range(self.get_total_num_seqs())]

    def get_current_seq_order(self) -> Sequence[int]:
        """
        :return: seq order
        """
        return self._seq_order


class Task12AXDataset(GeneratingDataset):
    """
    12AX memory task.
    This is a simple memory task where there is an outer loop and an inner loop.
    Description here: https://psych.colorado.edu/~oreilly/pubs-abstr.html#OReillyFrank06
    """

    _input_classes = "123ABCXYZ"  # noqa
    _output_classes = "LR"
    _getnewargs_exclude_attrs = Dataset._getnewargs_exclude_attrs.union(("input_dim", "output_dim"))

    def __init__(self, **kwargs):
        super(Task12AXDataset, self).__init__(
            input_dim=len(self._input_classes), output_dim=len(self._output_classes), **kwargs
        )

    def get_random_seq_len(self):
        """
        :rtype: int
        """
        return self.random.randint(10, 100)

    def generate_input_seq(self, seq_len):
        """
        Somewhat made up probability distribution.
        Try to make in a way that at least some "R" will occur in the output seq.
        Otherwise, "R"s are really rare.

        :param int seq_len:
        :rtype: list[int]
        """
        seq = self.random.choice(["", "1", "2"])
        while len(seq) < seq_len:
            if self.random.uniform() < 0.5:
                seq += self.random.choice(list("12"))
            if self.random.uniform() < 0.9:
                seq += self.random.choice(["AX", "BY"])
            while self.random.uniform() < 0.5:
                seq += self.random.choice(list(self._input_classes))
        return list(map(self._input_classes.index, seq[:seq_len]))

    @classmethod
    def make_output_seq(cls, input_seq):
        """
        :type input_seq: list[int]
        :rtype: list[int]
        """
        outer_state = ""
        inner_state = ""
        input_classes = cls._input_classes
        output_seq_str = ""
        for i in input_seq:
            c = input_classes[i]
            o = "L"
            if c in "12":
                outer_state = c
            elif c in "AB":
                inner_state = c
            elif c in "XY":
                if outer_state + inner_state + c in ["1AX", "2BY"]:
                    o = "R"
                inner_state = ""
            # Ignore other cases, "3CZ".
            output_seq_str += o
        return list(map(cls._output_classes.index, output_seq_str))

    def estimate_output_class_priors(self, num_trials, seq_len=10):
        """
        :type num_trials: int
        :param int seq_len:
        :rtype: (float, float)
        """
        count_l, count_r = 0, 0
        for i in range(num_trials):
            input_seq = self.generate_input_seq(seq_len)
            output_seq = self.make_output_seq(input_seq)
            count_l += output_seq.count(0)
            count_r += output_seq.count(1)
        return float(count_l) / (num_trials * seq_len), float(count_r) / (num_trials * seq_len)

    def generate_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        seq_len = self.get_random_seq_len()
        input_seq = self.generate_input_seq(seq_len)
        output_seq = self.make_output_seq(input_seq)
        features = class_idx_seq_to_1_of_k(input_seq, num_classes=len(self._input_classes))
        targets = numpy.array(output_seq, dtype="int32")
        return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


class TaskEpisodicCopyDataset(GeneratingDataset):
    """
    Episodic Copy memory task.
    This is a simple memory task where we need to remember a sequence.
    Described in: https://arxiv.org/abs/1511.06464
    Also tested for Associative LSTMs.
    This is a variant where the lengths are random, both for the chars and for blanks.
    """

    # Blank, delimiter and some chars.
    _input_classes = " .01234567"
    _output_classes = _input_classes
    _getnewargs_exclude_attrs = Dataset._getnewargs_exclude_attrs.union(("input_dim", "output_dim"))

    def __init__(self, **kwargs):
        super(TaskEpisodicCopyDataset, self).__init__(
            input_dim=len(self._input_classes), output_dim=len(self._output_classes), **kwargs
        )

    def generate_input_seq(self):
        """
        :rtype: list[int]
        """
        seq = ""
        # Start with random chars.
        rnd_char_len = self.random.randint(1, 10)
        seq += "".join([self.random.choice(list(self._input_classes[2:])) for _ in range(rnd_char_len)])
        blank_len = self.random.randint(1, 100)
        seq += " " * blank_len  # blanks
        seq += "."  # 1 delim
        seq += "." * (rnd_char_len + 1)  # we wait for the outputs + 1 delim
        return list(map(self._input_classes.index, seq))

    @classmethod
    def make_output_seq(cls, input_seq):
        """
        :type input_seq: list[int]
        :rtype: list[int]
        """
        input_classes = cls._input_classes
        input_mem = ""
        output_seq_str = ""
        state = 0
        for i in input_seq:
            c = input_classes[i]
            if state == 0:
                output_seq_str += " "
                if c == " ":
                    pass  # just ignore
                elif c == ".":
                    state = 1  # start with recall now
                else:
                    input_mem += c
            else:  # recall from memory
                # Ignore input.
                if not input_mem:
                    output_seq_str += "."
                else:
                    output_seq_str += input_mem[:1]
                    input_mem = input_mem[1:]
        return list(map(cls._output_classes.index, output_seq_str))

    def generate_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        input_seq = self.generate_input_seq()
        output_seq = self.make_output_seq(input_seq)
        features = class_idx_seq_to_1_of_k(input_seq, num_classes=len(self._input_classes))
        targets = numpy.array(output_seq)
        return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


class TaskXmlModelingDataset(GeneratingDataset):
    """
    XML modeling memory task.
    This is a memory task where we need to remember a stack.
    Defined in Jozefowicz et al. (2015).
    Also tested for Associative LSTMs.
    """

    # Blank, XML-tags and some chars.
    _input_classes = " <>/abcdefgh"  # noqa
    _output_classes = _input_classes
    _getnewargs_exclude_attrs = Dataset._getnewargs_exclude_attrs.union(("input_dim", "output_dim"))

    def __init__(self, limit_stack_depth=4, **kwargs):
        super(TaskXmlModelingDataset, self).__init__(
            input_dim=len(self._input_classes), output_dim=len(self._output_classes), **kwargs
        )
        self.limit_stack_depth = limit_stack_depth

    def generate_input_seq(self):
        """
        :rtype: list[int]
        """
        # Because this is a prediction task, start with blank,
        # and the output seq should predict the next char after the blank.
        seq = " "
        xml_stack = []
        while True:
            if not xml_stack or (len(xml_stack) < self.limit_stack_depth and self.random.rand() > 0.6):
                tag_len = self.random.randint(1, 10)
                tag = "".join([self.random.choice(list(self._input_classes[4:])) for _ in range(tag_len)])
                seq += "<%s>" % tag
                xml_stack += [tag]
            else:
                seq += "</%s>" % xml_stack.pop()
            if not xml_stack and self.random.rand() > 0.2:
                break
        return list(map(self._input_classes.index, seq))

    @classmethod
    def make_output_seq(cls, input_seq):
        """
        :type input_seq: list[int]
        :rtype: list[int]
        """
        input_seq_str = "".join(cls._input_classes[i] for i in input_seq)
        xml_stack = []
        output_seq_str = ""
        state = 0
        for c in input_seq_str:
            if c in " >":
                output_seq_str += "<"  # We expect an open char.
                assert state != 1, repr(input_seq_str)
                state = 1  # expect beginning of tag
            elif state == 1:  # in beginning of tag
                output_seq_str += " "  # We don't know yet.
                assert c == "<", repr(input_seq_str)
                state = 2
            elif state == 2:  # first char in tag
                if c == "/":
                    assert xml_stack, repr(input_seq_str)
                    output_seq_str += xml_stack[-1][0]
                    xml_stack[-1] = xml_stack[-1][1:]
                    state = 4  # closing tag
                else:  # opening tag
                    output_seq_str += " "  # We don't know yet.
                    assert c not in " <>/", repr(input_seq_str)
                    state = 3
                    xml_stack += [c]
            elif state == 3:  # opening tag
                output_seq_str += " "  # We don't know.
                xml_stack[-1] += c
            elif state == 4:  # closing tag
                assert xml_stack, repr(input_seq_str)
                if not xml_stack[-1]:
                    output_seq_str += ">"
                    xml_stack.pop()
                    state = 0
                else:
                    output_seq_str += xml_stack[-1][0]
                    xml_stack[-1] = xml_stack[-1][1:]
            else:
                assert False, "invalid state %i. input %r" % (state, input_seq_str)
        return list(map(cls._output_classes.index, output_seq_str))

    def generate_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        input_seq = self.generate_input_seq()
        output_seq = self.make_output_seq(input_seq)
        features = class_idx_seq_to_1_of_k(input_seq, num_classes=len(self._input_classes))
        targets = numpy.array(output_seq)
        return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


class TaskVariableAssignmentDataset(GeneratingDataset):
    """
    Variable Assignment memory task.
    This is a memory task to test for key-value retrieval.
    Defined in Associative LSTM paper.
    """

    # Blank/Delim/End, Store/Query, and some chars for key/value.
    _input_classes = " ,.SQ()abcdefgh"  # noqa
    _output_classes = _input_classes
    _getnewargs_exclude_attrs = Dataset._getnewargs_exclude_attrs.union(("input_dim", "output_dim"))

    def __init__(self, **kwargs):
        super(TaskVariableAssignmentDataset, self).__init__(
            input_dim=len(self._input_classes), output_dim=len(self._output_classes), **kwargs
        )

    def generate_input_seq(self):
        """
        :rtype: list[int]
        """
        seq = ""
        from collections import OrderedDict

        store = OrderedDict()
        # First the assignments.
        num_assignments = self.random.randint(1, 5)
        for i in range(num_assignments):
            key_len = self.random.randint(2, 5)
            while True:  # find unique key
                key = "".join([self.random.choice(list(self._input_classes[7:])) for _ in range(key_len)])
                if key not in store:
                    break
            value_len = self.random.randint(1, 2)
            value = "".join([self.random.choice(list(self._input_classes[7:])) for _ in range(value_len)])
            if seq:
                seq += ","
            seq += "S(%s,%s)" % (key, value)
            store[key] = value
        # Now one query.
        key = self.random.choice(store.keys())
        value = store[key]
        seq += ",Q(%s)" % key
        seq += "%s." % value
        return list(map(self._input_classes.index, seq))

    @classmethod
    def make_output_seq(cls, input_seq):
        """
        :type input_seq: list[int]
        :rtype: list[int]
        """
        input_seq_str = "".join(cls._input_classes[i] for i in input_seq)
        store = {}
        key, value = "", ""
        output_seq_str = ""
        state = 0
        for c in input_seq_str:
            if state == 0:
                key = ""
                if c == "S":
                    state = 1  # store
                elif c == "Q":
                    state = 2  # query
                elif c in " ,":
                    pass  # can be ignored
                else:
                    assert False, "c %r in %r" % (c, input_seq_str)
                output_seq_str += " "
            elif state == 1:  # store
                assert c == "(", repr(input_seq_str)
                state = 1.1
                output_seq_str += " "
            elif state == 1.1:  # store.key
                if c == ",":
                    assert key
                    value = ""
                    state = 1.5  # store.value
                else:
                    assert c not in " .,SQ()", repr(input_seq_str)
                    key += c
                output_seq_str += " "
            elif state == 1.5:  # store.value
                if c == ")":
                    assert value
                    store[key] = value
                    state = 0
                else:
                    assert c not in " .,SQ()", repr(input_seq_str)
                    value += c
                output_seq_str += " "
            elif state == 2:  # query
                assert c == "(", repr(input_seq_str)
                state = 2.1
                output_seq_str += " "
            elif state == 2.1:  # query.key
                if c == ")":
                    value = store[key]
                    output_seq_str += value[0]
                    value = value[1:]
                    state = 2.5
                else:
                    assert c not in " .,SQ()", repr(input_seq_str)
                    key += c
                    output_seq_str += " "
            elif state == 2.5:  # query result
                assert c not in " .,SQ()", repr(input_seq_str)
                if value:
                    output_seq_str += value[0]
                    value = value[1:]
                else:
                    output_seq_str += "."
                    state = 2.6
            elif state == 2.6:  # query result end
                assert c == ".", repr(input_seq_str)
                output_seq_str += " "
            else:
                assert False, "invalid state %i, input %r" % (state, input_seq_str)
        return list(map(cls._output_classes.index, output_seq_str))

    def generate_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        input_seq = self.generate_input_seq()
        output_seq = self.make_output_seq(input_seq)
        features = class_idx_seq_to_1_of_k(input_seq, num_classes=len(self._input_classes))
        targets = numpy.array(output_seq)
        return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


class TaskNumberBaseConvertDataset(GeneratingDataset):
    """
    Task: E.g: Get some number in octal and convert it to binary (e.g. "10101001").
    Or basically convert some number from some base into another base.
    """

    _getnewargs_exclude_attrs = Dataset._getnewargs_exclude_attrs.union(("input_dim", "output_dim"))

    def __init__(self, input_base=8, output_base=2, min_input_seq_len=1, max_input_seq_len=8, **kwargs):
        """
        :param int input_base:
        :param int output_base:
        :param int min_input_seq_len:
        :param int max_input_seq_len:
        """
        super(TaskNumberBaseConvertDataset, self).__init__(
            input_dim=input_base, output_dim={"data": (input_base, 1), "classes": (output_base, 1)}, **kwargs
        )
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"  # noqa
        assert 2 <= input_base <= len(chars) and 2 <= output_base <= len(chars)
        self.input_base = input_base
        self.output_base = output_base
        self._input_classes = chars[:input_base]
        self._output_classes = chars[:output_base]
        self.labels = {"data": self._input_classes, "classes": self._output_classes}
        assert 0 < min_input_seq_len <= max_input_seq_len
        self.min_input_seq_len = min_input_seq_len
        self.max_input_seq_len = max_input_seq_len

    def get_random_input_seq_len(self):
        """
        :rtype: int
        """
        return self.random.randint(self.min_input_seq_len, self.max_input_seq_len + 1)

    def generate_input_seq(self):
        """
        :rtype: list[int]
        """
        seq_len = self.get_random_input_seq_len()
        seq = [self.random.randint(0, len(self._input_classes)) for _ in range(seq_len)]
        return seq

    def make_output_seq(self, input_seq):
        """
        :param list[int] input_seq:
        :rtype: list[int]
        """
        number = 0
        for i, d in enumerate(reversed(input_seq)):
            number += d * (self.input_base**i)
        output_seq = []
        while number:
            output_seq.insert(0, number % self.output_base)
            number //= self.output_base
        if not output_seq:
            output_seq = [0]
        return output_seq

    def generate_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        input_seq = self.generate_input_seq()
        output_seq = self.make_output_seq(input_seq)
        features = numpy.array(input_seq)
        targets = numpy.array(output_seq)
        return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


class DummyDataset(GeneratingDataset):
    """
    Some dummy data, which does not have any meaning.
    If you want to have artificial data with some meaning, look at other datasets here.
    The input are some dense data, the outputs are sparse.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_seqs,
        seq_len=2,
        input_max_value=10.0,
        input_shift=None,
        input_scale=None,
        **kwargs,
    ):
        """
        :param int|None input_dim:
        :param int|dict[str,int|(int,int)|dict] output_dim:
        :param int|float num_seqs:
        :param int|dict[str,int] seq_len:
        :param float input_max_value:
        :param float|None input_shift:
        :param float|None input_scale:
        """
        super(DummyDataset, self).__init__(input_dim=input_dim, output_dim=output_dim, num_seqs=num_seqs, **kwargs)
        self.seq_len = seq_len
        self.input_max_value = input_max_value
        if input_shift is None:
            input_shift = -input_max_value / 2.0
        self.input_shift = input_shift
        if input_scale is None:
            input_scale = 1.0 / self.input_max_value
        self.input_scale = input_scale

    def generate_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        seq_len = self.seq_len
        i1 = seq_idx
        i2 = i1 + seq_len * self.num_inputs
        features = numpy.array(
            [((i % self.input_max_value) + self.input_shift) * self.input_scale for i in range(i1, i2)]
        ).reshape((seq_len, self.num_inputs))
        i1, i2 = i2, i2 + seq_len
        targets = numpy.array([i % self.num_outputs["classes"][0] for i in range(i1, i2)])
        return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


class DummyDatasetMultipleSequenceLength(DummyDataset):
    """
    Like :class:`DummyDataset` but has provides seqs with different sequence lengths.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_seqs,
        seq_len=None,
        input_max_value=10.0,
        input_shift=None,
        input_scale=None,
        **kwargs,
    ):
        """
        :param int input_dim:
        :param int output_dim:
        :param int|float num_seqs:
        :param int|dict[str,int] seq_len:
        :param float input_max_value:
        :param float|None input_shift:
        :param float|None input_scale:
        """
        if seq_len is None:
            seq_len = {"data": 10, "classes": 20}
        super(DummyDatasetMultipleSequenceLength, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_seqs=num_seqs,
            seq_len=seq_len,
            input_max_value=input_max_value,
            input_shift=input_shift,
            input_scale=input_scale,
            **kwargs,
        )

    def generate_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        assert isinstance(self.seq_len, dict)
        seq_len_data = self.seq_len["data"]
        seq_len_classes = self.seq_len["classes"]
        i1 = seq_idx
        i2 = i1 + seq_len_data * self.num_inputs
        features = numpy.array(
            [((i % self.input_max_value) + self.input_shift) * self.input_scale for i in range(i1, i2)]
        ).reshape((seq_len_data, self.num_inputs))
        i1, i2 = i2, i2 + seq_len_classes
        targets = numpy.array([i % self.num_outputs["classes"][0] for i in range(i1, i2)])
        return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


class DummyDatasetMultipleDataKeys(DummyDataset):
    """
    Like :class:`DummyDataset` this class provides dummy data without any meaning.
    But it extends :class:`DummyDataset` such that it is able to provide data for multiple data keys,
    not only `"data"` and `"classes"` (those are also overridable, though the current implementation
    expects a `"data"` key).
    Further, `output_dim` is expected to be a `dict` now, which defines the data format for each
    data key, which also enables the user to customize whether the data is sparse or dense.
    It also provides the function of :class:`DummyDatasetMultipleSequenceLength` to customize the
    sequence length for each data point.
    """

    _getnewargs_exclude_attrs = Dataset._getnewargs_exclude_attrs.union(("input_dim",))

    def __init__(
        self,
        output_dim,
        num_seqs,
        seq_len=None,
        input_max_value=10.0,
        input_shift=None,
        input_scale=None,
        data_keys=None,
        **kwargs,
    ):
        """
        :param dict[str,int|(int,int)|dict] output_dim: `dict` defining the output for each data key
          (e.g. `{"data": [200, 2], "classes": [100, 1]}`).
        :param int|float num_seqs:
        :param int|dict[str,int] seq_len: definition of the sequence length for each data key,
          if `int` the given length is used for all data keys.
        :param float input_max_value:
        :param float|None input_shift:
        :param float|None input_scale:
        :param list[str]|None data_keys: explicit declaration of the data keys,
          if `None` `"data"` and `"classes"` are used.
        """
        if data_keys is None:
            data_keys = ["data", "classes"]
        self.data_keys = data_keys

        _seq_len = 20
        if isinstance(seq_len, int):
            _seq_len = seq_len
            seq_len = None
        if seq_len is None:
            seq_len = {}
            for key in self.data_keys:
                seq_len[key] = _seq_len
        assert set(data_keys) == set(
            seq_len.keys()
        ), "%s: the keys of seq_len (%s) must match the keys in data_keys=%s." % (
            self,
            str(seq_len.keys()),
            str(data_keys),
        )
        assert isinstance(
            output_dim, dict
        ), "%s: output_dim %r must be a dict containing a definition for each key in data_keys." % (self, output_dim)
        assert set(data_keys) == set(
            output_dim.keys()
        ), "%s: the keys of output_dim (%s) must match the keys in data_keys=%s." % (
            self,
            str(output_dim.keys()),
            str(data_keys),
        )

        super(DummyDatasetMultipleDataKeys, self).__init__(
            input_dim=None,  # this was only used for the definition of "data", but this is handled by `output_dim` now.
            output_dim=output_dim,
            num_seqs=num_seqs,
            seq_len=seq_len,
            input_max_value=input_max_value,
            input_shift=input_shift,
            input_scale=input_scale,
            **kwargs,
        )

    def generate_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        features = {}
        i1 = seq_idx

        for key in self.data_keys:
            seq_len = self.seq_len[key]
            output_dim = self.num_outputs[key][0]
            is_sparse = self.num_outputs[key][1] == 1

            if is_sparse:
                i2 = i1 + seq_len
                features[key] = numpy.array([i % self.num_outputs[key][0] for i in range(i1, i2)])
            else:
                i2 = i1 + seq_len * output_dim
                features[key] = numpy.array(
                    [((i % self.input_max_value) + self.input_shift) * self.input_scale for i in range(i1, i2)]
                ).reshape((seq_len, output_dim))
            i1 = i2

        return DatasetSeq(seq_idx=seq_idx, features=features, targets=None)


class StaticDataset(CachedDataset2):
    """
    Provide all the data as a list of dict of numpy arrays.
    """

    @classmethod
    def copy_from_dataset(cls, dataset, start_seq_idx=0, max_seqs=None):
        """
        :param Dataset dataset:
        :param int start_seq_idx:
        :param int|None max_seqs:
        :rtype: StaticDataset
        """
        if isinstance(dataset, StaticDataset):
            return cls(
                data=dataset.data,
                target_list=dataset.target_list,
                output_dim=dataset.num_outputs,
                input_dim=dataset.num_inputs,
            )
        seq_idx = start_seq_idx
        data = []
        while dataset.is_less_than_num_seqs(seq_idx):
            dataset.load_seqs(seq_idx, seq_idx + 1)
            if max_seqs is not None and len(data) >= max_seqs:
                break
            seq_data = {key: dataset.get_data(seq_idx, key) for key in dataset.get_data_keys()}
            data.append(seq_data)
            seq_idx += 1
        return cls(
            data=data,
            target_list=dataset.get_target_list(),
            output_dim=dataset.num_outputs,
            input_dim=dataset.num_inputs,
        )

    def __init__(self, data, target_list=None, input_dim=None, output_dim=None, **kwargs):
        """
        :param list[dict[str,numpy.ndarray]] data: list of seqs, each provide the data for each data-key
        :param target_list:
        :param int|None input_dim:
        :param int|dict[str,(int,int)|list[int]] output_dim:
        """
        super(StaticDataset, self).__init__(**kwargs)

        assert len(data) > 0
        self.data = data
        first_data = data[0]
        self.data_keys = sorted(first_data.keys())
        if "data" in self.data_keys:
            self.data_keys.remove("data")
            self.data_keys.insert(0, "data")
        if target_list is not None:
            for key in target_list:
                assert key in self.data_keys
        else:
            target_list = list(self.data_keys)
            if "data" in target_list:
                target_list.remove("data")
        self.target_list = target_list

        if output_dim is None:
            output_dim = {}
        output_dim = convert_data_dims(output_dim, leave_dict_as_is=False)
        if input_dim is not None and "data" not in output_dim:
            assert "data" in self.data_keys
            output_dim["data"] = (input_dim, 2)  # assume dense, not sparse
        for key, value in first_data.items():
            if key not in output_dim:
                output_dim[key] = (value.shape[-1] if value.ndim >= 2 else 0, len(value.shape))
        if input_dim is None and "data" in self.data_keys:
            input_dim = output_dim["data"][0]
        for key in self.data_keys:
            first_data_output = first_data[key]
            assert key in output_dim
            assert output_dim[key][1] == len(first_data_output.shape)
            if len(first_data_output.shape) >= 2:
                assert output_dim[key][0] == first_data_output.shape[-1]
        assert set(output_dim.keys()) == set(self.data_keys), "output_dim does not match the given data"

        self.num_inputs = input_dim
        output_dim = convert_data_dims(output_dim, leave_dict_as_is=False)
        if "data" not in output_dim and input_dim is not None:
            output_dim["data"] = (input_dim, 2)  # not sparse
        self.num_outputs = output_dim

        self._seq_order = None
        self.init_seq_order(epoch=1)

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """
        :param int|None epoch:
        :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
        :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order. Only possible
          if the dataset has such indices (see self.have_corpus_seq_idx()).
        :rtype: bool
        :returns whether the order changed (True is always safe to return)
        """
        super(StaticDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
        if seq_order is not None:
            assert not seq_list
            self._seq_order = seq_order
        elif seq_list is not None:
            import re

            # If the re.match fails here, some seq tag is invalid.
            self._seq_order = [int(re.match("^seq-(\\d+)$", seq).group(1)) for seq in seq_list]
        else:
            default_data_key = self.data_keys[0]
            self._seq_order = self.get_seq_order_for_epoch(
                epoch=epoch, num_seqs=len(self.data), get_seq_len=lambda i: self.data[i][default_data_key].shape[0]
            )
        self._num_seqs = len(self._seq_order)
        return True

    def _collect_single_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        corpus_seq_idx = self._seq_order[seq_idx]
        data = self.data[corpus_seq_idx]
        return DatasetSeq(
            seq_idx=seq_idx, seq_tag="seq-%i" % corpus_seq_idx, features={key: data[key] for key in self.data_keys}
        )

    def get_data_keys(self):
        """
        :rtype: list[str]
        """
        return self.data_keys

    def get_target_list(self):
        """
        :rtype: list[str]
        """
        return self.target_list

    def get_data_dtype(self, key):
        """
        :param str key:
        :rtype: str
        """
        return self.data[0][key].dtype

    def get_total_num_seqs(self):
        """
        :rtype: int
        """
        return len(self.data)

    def get_all_tags(self):
        """
        :return: list of all seq tags, of the whole dataset, without partition epoch.
        :rtype: list[str]
        """
        return ["seq-%i" % i for i in range(self.get_total_num_seqs())]

    def get_tag(self, sorted_seq_idx):
        """
        :param int sorted_seq_idx:
        :rtype: str
        """
        return "seq-%i" % self.get_corpus_seq_idx(sorted_seq_idx)

    def have_corpus_seq_idx(self):
        """
        :rtype: bool
        :return: whether you can call self.get_corpus_seq_idx()
        """
        return True

    def get_corpus_seq_idx(self, seq_idx):
        """
        :param int seq_idx: sorted sequence index from the current epoch, depending on seq_ordering
        :return: the sequence index as-is in the original corpus (as if you would have sorting="default").
        :rtype: int
        """
        return self._seq_order[seq_idx]


class CopyTaskDataset(GeneratingDataset):
    """
    Copy task.
    Input/output is exactly the same random sequence of sparse labels.
    """

    _getnewargs_exclude_attrs = Dataset._getnewargs_exclude_attrs.union(("input_dim", "output_dim"))

    def __init__(self, nsymbols, minlen=0, maxlen=0, minlen_epoch_factor=0, maxlen_epoch_factor=0, **kwargs):
        """
        :param int nsymbols:
        :param int minlen:
        :param int maxlen:
        :param float minlen_epoch_factor:
        :param float maxlen_epoch_factor:
        """
        # Sparse data.
        super(CopyTaskDataset, self).__init__(
            input_dim=nsymbols, output_dim={"data": (nsymbols, 1), "classes": (nsymbols, 1)}, **kwargs
        )

        assert nsymbols <= 256
        self.nsymbols = nsymbols
        self.minlen = minlen
        self.maxlen = maxlen
        self.minlen_epoch_factor = minlen_epoch_factor
        self.maxlen_epoch_factor = maxlen_epoch_factor

    def get_random_seq_len(self):
        """
        :rtype: int
        """
        assert isinstance(self.epoch, int)
        minlen = int(self.minlen + self.minlen_epoch_factor * self.epoch)
        maxlen = int(self.maxlen + self.maxlen_epoch_factor * self.epoch)
        assert 0 < minlen <= maxlen
        return self.random.randint(minlen, maxlen + 1)

    def generate_seq(self, seq_idx):
        """
        :type seq_idx: int
        :rtype: DatasetSeq
        """
        seq_len = self.get_random_seq_len()
        seq = [self.random.randint(0, self.nsymbols) for _ in range(seq_len)]
        seq_np = numpy.array(seq, dtype="int8")
        return DatasetSeq(seq_idx=seq_idx, features=seq_np, targets={"classes": seq_np})
