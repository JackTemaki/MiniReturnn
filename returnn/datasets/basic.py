"""
This defines the base dataset class :class:`Dataset`.
"""

from __future__ import annotations

__author__ = "Patrick Doetsch"
__copyright__ = "Copyright 2015"
__credits__ = ["Patrick Doetsch", "Paul Voigtlaender"]
__license__ = "RWTHASR"
__version__ = "0.9"
__maintainer__ = "Patrick Doetsch"
__email__ = "doetsch@i6.informatik.rwth-aachen.de"

from threading import RLock
from random import Random
import numpy
import functools
import typing
from typing import Optional, Union, Type, Dict, List

from returnn.datasets.util.vocabulary import Vocabulary
from returnn.util.basic import try_run, NumbersDict, OptionalNotImplementedError


class Dataset(object):
    """
    Base class for any dataset. This defines the dataset API.
    """

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        :type config: returnn.config.Config
        :param dict[str] kwargs: passed on to __init__
        :rtype: Dataset
        """
        return cls(**kwargs)

    def __init__(
        self,
        name=None,
        seq_ordering="default",
        fixed_random_seed=None,
        random_seed_offset=None,
        partition_epoch=None,
        repeat_epoch=None,
        seq_list_filter_file=None,
        unique_seq_tags=False,
        seq_order_seq_lens_file=None,
        shuffle_frames_of_nseqs=0,
        estimated_num_seqs=None,
    ):
        """
        :param str name: e.g. "train" or "eval"
        :param int window: features will be of dimension window * feature_dim, as we add a context-window around.
          not all datasets support this option.
        :param str seq_ordering: "batching"-option in config. e.g. "default", "sorted" or "random".
          See self.get_seq_order_for_epoch() for more details.
        :param int|None fixed_random_seed: for the shuffling, e.g. for seq_ordering='random'.
            otherwise epoch will be used.
            useful when used as eval dataset.
        :param int|None random_seed_offset: for shuffling, e.g. for seq_ordering='random'.
            ignored when fixed_random_seed is set.
        :param int|None partition_epoch:
        :param int|None repeat_epoch: Repeat the sequences in an epoch this many times. Useful to scale the dataset
          relative to other datasets, e.g. when used in CombinedDataset. Not allowed to be used in combination with
          partition_epoch.
        :param str|None seq_list_filter_file: defines a subset of sequences (by tag) to use
        :param bool unique_seq_tags: uniquify seqs with same seq tags in seq order
        :param str|None seq_order_seq_lens_file: for seq order, use the seq length given by this file
        :param int shuffle_frames_of_nseqs: shuffles the frames. not always supported
        :param None|int estimated_num_seqs: for progress reporting in case the real num_seqs is unknown
        """
        self.name = name or ("dataset_id%s" % id(self))
        self.lock = None  # type: Optional[RLock]  # Used when manipulating our data potentially from multiple threads.
        self.rnd_seq_drop = None  # type: typing.Optional[Random]
        self.num_inputs = 0  # usually not used, but num_outputs instead, which is more generic
        self.num_outputs = (
            None
        )  # type: typing.Optional[typing.Dict[str,typing.Tuple[int,int]]]  # tuple is num-classes, len(shape).  # nopep8
        self.seq_ordering = seq_ordering  # "default", "sorted" or "random". See self.get_seq_order_for_epoch().
        self.fixed_random_seed = fixed_random_seed
        if random_seed_offset is None:
            random_seed_offset = self._get_default_random_seed_offset()
        self.random_seed_offset = random_seed_offset
        self.partition_epoch = partition_epoch or 1
        self.repeat_epoch = repeat_epoch or 1
        self._seq_list_filter_file = seq_list_filter_file
        self.seq_tags_filter = set(self._load_seq_list_file(seq_list_filter_file)) if seq_list_filter_file else None
        self.unique_seq_tags = unique_seq_tags
        self._seq_order_seq_lens_file = seq_order_seq_lens_file
        self._seq_order_seq_lens_by_idx = None
        # There is probably no use case for combining the two, so avoid potential misconfiguration.
        assert (
            self.partition_epoch == 1 or self.repeat_epoch == 1
        ), "Combining partition_epoch and repeat_epoch is prohibited."
        self.labels = {}  # type: typing.Dict[str,typing.List[str]]
        self.weights = {}
        self._num_timesteps = 0
        self._num_seqs = 0
        self._estimated_num_seqs = estimated_num_seqs
        self.shuffle_frames_of_nseqs = shuffle_frames_of_nseqs
        self.epoch = None

    def __repr__(self):
        return "<%s %r epoch=%s>" % (
            self.__class__.__name__,
            getattr(self, "name", "<unknown>"),
            getattr(self, "epoch", "<unknown>"),
        )

    _getnewargs_exclude_attrs = set()  # type: typing.Set[str]

    @staticmethod
    def _create_from_reduce(cls, kwargs, state) -> Dataset:
        """
        :param type cls:
        :param dict[str] kwargs:
        :param dict[str] state:
        :rtype: Dataset
        """
        assert issubclass(cls, Dataset)
        ds = cls(**kwargs)
        for attr, value in state.items():
            setattr(ds, attr, value)
        return ds

    def __reduce__(self):
        import inspect

        kwargs = {}
        for cls in self.__class__.__mro__:
            if isinstance(cls, type) and issubclass(cls, Dataset):
                for arg in inspect.getargs(cls.__init__.__code__).args[1:]:
                    if arg in self._getnewargs_exclude_attrs:
                        continue
                    if hasattr(self, "_" + arg):
                        kwargs[arg] = getattr(self, "_" + arg)
                    else:
                        kwargs[arg] = getattr(self, arg)

        state = {attr: getattr(self, attr) for attr in ["epoch"]}
        return Dataset._create_from_reduce, (self.__class__, kwargs, state)

    @staticmethod
    def _get_default_random_seed_offset():
        """
        :return: 0 usually
        :rtype: int
        """
        from returnn.config import get_global_config

        config = get_global_config(raise_exception=False)
        if not config:
            return 0
        if config.is_true("use_horovod"):
            import returnn.tf.horovod

            if returnn.tf.horovod.get_ctx().is_dataset_distribution_random_seed_offset():
                return returnn.tf.horovod.get_ctx().rank() * 16127
        return 0

    @staticmethod
    def _load_seq_list_file(filename, use_cache_manager=False, expect_list=True):
        """
        :param str filename:
        :param bool use_cache_manager:
        :param bool expect_list:
        :rtype: list[str]|dict[str,list[str]]
        """
        if use_cache_manager:
            import returnn.util.basic

            filename = returnn.util.basic.cf(filename)
        if filename.endswith(".pkl"):
            import pickle

            seq_list = pickle.load(open(filename, "rb"))
            if expect_list:
                assert isinstance(seq_list, list)
        elif filename.endswith(".gz"):
            import gzip

            seq_list = gzip.open(filename, "rt").read().splitlines()
        else:
            seq_list = open(filename).read().splitlines()
        return seq_list

    def is_cached(self, start, end):
        """
        :param int start: like in load_seqs(), sorted seq idx
        :param int end: like in load_seqs(), sorted seq idx
        :rtype: bool
        :returns whether we have the full range (start,end) of sorted seq idx.
        """
        if start == end:
            return True  # Empty.
        assert start < end
        return False

    def get_seq_length(self, seq_idx: int) -> NumbersDict:
        """
        :param seq_idx:
        :returns the len of the input features and the len of the target sequence.
        """
        raise NotImplementedError

    def get_estimated_seq_length(self, seq_idx):
        """
        In contrast to self.get_seq_length(),
        this method is designed to work for sequences that have not been loaded yet
        via self.load_seqs().
        Used by meta-datasets for sequence ordering.
        Currently we only provide one number, i.e. do not give different
        estimates for the different data keys (as in get_seq_length()).
        It is up to the dataset what this number represents
        and how it is computed.

        :param int seq_idx: for current epoch, not the corpus seq idx
        :rtype: int
        :returns sequence length estimate (for sorting)
        """
        raise OptionalNotImplementedError

    def get_num_timesteps(self):
        """
        :rtype: int
        """
        assert self._num_timesteps > 0
        return self._num_timesteps

    def load_seqs(self, start, end):
        """
        Load data sequences, such that self.get_data() & friends can return the data.

        :param int start: start sorted seq idx, inclusive
        :param int end: end sorted seq idx, exclusive
        """
        assert start >= 0
        assert start <= end
        if self.is_cached(start, end):
            return

        if self.shuffle_frames_of_nseqs > 0:
            # We always load N seqs at once and shuffle all their frames.
            start, end = self._get_load_seqs_superset(start, end)
            self._load_seqs(start, end)
            while start < end:
                self._shuffle_frames_in_seqs(start, start + self.shuffle_frames_of_nseqs)
                start += self.shuffle_frames_of_nseqs
        else:
            self._load_seqs(start, end)

    def _get_load_seqs_superset(self, start, end):
        """
        :type start: int
        :type end: int
        :returns the superset (start,end) of seqs to be loaded.
        For shuffle_frames_of_nseqs > 0, we always load N seqs at once
        and shuffle all their frames,
        thus start/end will be aligned to self.shuffle_frames_of_nseqs.
        """
        assert start <= end
        assert start < self.num_seqs
        if self.shuffle_frames_of_nseqs > 0:
            m = self.shuffle_frames_of_nseqs
            start -= start % m
            end += (m - (end % m)) % m
        return start, end

    def _shuffle_frames_in_seqs(self, start, end):
        raise OptionalNotImplementedError

    def _load_seqs(self, start, end):
        """
        Load data sequences.
        If end > num_seqs, will not load them.

        :param int start: inclusive seq idx start
        :param int end: exclusive seq idx end. can be more than num_seqs
        """
        raise NotImplementedError

    def _get_seq_order_seq_lens_by_idx(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: int
        """
        if not self._seq_order_seq_lens_by_idx:
            assert self._seq_order_seq_lens_file
            if self._seq_order_seq_lens_file.endswith(".gz"):
                import gzip

                raw = gzip.GzipFile(self._seq_order_seq_lens_file, "rb").read()
            else:
                raw = open(self._seq_order_seq_lens_file, "rb").read()
            seq_lens = eval(raw)
            assert isinstance(seq_lens, dict)
            all_tags = self.get_all_tags()
            self._seq_order_seq_lens_by_idx = [seq_lens[tag] for tag in all_tags]
        return self._seq_order_seq_lens_by_idx[seq_idx]

    def get_seq_order_for_epoch(self, epoch, num_seqs, get_seq_len=None):
        """
        Returns the order of the given epoch.
        This is mostly a static method, except that is depends on the configured type of ordering,
        such as 'default' (= as-is), 'sorted' or 'random'. 'sorted' also uses the sequence length.

        :param int epoch: for 'random', this determines the random seed
        :param int num_seqs:
        :param ((int) -> int)|None get_seq_len: function (originalSeqIdx: int) -> int
        :return: the order for the given epoch. such that seq_idx -> underlying idx
        :rtype: typing.Sequence[int]
        """
        partition_epoch = self.partition_epoch or 1
        repeat_epoch = self.repeat_epoch or 1
        assert num_seqs > 0
        # Make sure it is a proper integer now.
        assert num_seqs == int(num_seqs)
        num_seqs = int(num_seqs)
        if self._seq_order_seq_lens_file:
            get_seq_len = self._get_seq_order_seq_lens_by_idx

        if self.seq_ordering == "default":
            seq_index = range(num_seqs)
        elif self.seq_ordering.startswith("default_every_n:"):
            # This order is useful if you have "initial_state": "keep_over_epoch",
            # where num == max_seqs, batch_size = inf, max_seq_len = inf, chunking = None.
            _, num = self.seq_ordering.split(":")
            num = int(num)
            seq_index = numpy.arange(num_seqs // num, dtype="int64").repeat(num)
            for i in range(1, num):
                seq_index[i::num] += i * (num_seqs // num)
        elif self.seq_ordering == "reverse":
            seq_index = range(num_seqs - 1, -1, -1)  # type: Union[range, typing.Sequence[int]]
        elif self.seq_ordering in ["sorted", "sorted_reverse"]:
            assert get_seq_len
            reverse = -1 if self.seq_ordering == "sorted_reverse" else 1
            seq_lens = [reverse * get_seq_len(i) for i in range(num_seqs)]
            seq_index = numpy.argsort(seq_lens, kind="stable")
        elif self.seq_ordering.startswith("random"):
            tmp = self.seq_ordering.split(":")
            nth = int(tmp[1]) if len(tmp) > 1 else 1
            # Keep this deterministic! Use fixed seed.
            rnd_seed = self._get_random_seed_for_epoch(epoch=epoch, num_epochs_fixed=nth)
            random_generator = numpy.random.RandomState(rnd_seed)
            seq_index = random_generator.permutation(num_seqs)
        elif self.seq_ordering.startswith("sort_bin_shuffle"):
            # Shuffle seqs, sort by length, and shuffle bins (then shuffle seqs within each bin if sort_bin_shuffle_x2).
            assert get_seq_len
            tmp = self.seq_ordering.split(":")[1:]
            # Keep this deterministic! Use fixed seed.
            if len(tmp) <= 1:
                nth = 1
            else:
                nth = int(tmp[1])
            rnd_seed = self._get_random_seed_for_epoch(epoch=epoch, num_epochs_fixed=nth)
            random_generator = numpy.random.RandomState(rnd_seed)
            seq_index = random_generator.permutation(num_seqs).tolist()  # type: Union[List[int], numpy.ndarray]
            seq_index.sort(key=get_seq_len)  # Sort by length, starting with shortest.
            if len(tmp) == 0:
                bins = 2
            else:
                if tmp[0].startswith("."):  # starting with "." -> approx chunk size (num of seqs in one bin)
                    bins = max(num_seqs // int(tmp[0][1:]), 2)
                else:  # the number of bins
                    bins = int(tmp[0])
            bin_ids = random_generator.permutation(bins)  # Shuffle bins.
            out_index = []
            for i in bin_ids:
                if i == bins - 1:
                    part = seq_index[i * len(seq_index) // bins :][:]
                else:
                    part = seq_index[i * len(seq_index) // bins : (i + 1) * len(seq_index) // bins][:]
                if self.seq_ordering.startswith("sort_bin_shuffle_x2"):
                    random_generator.shuffle(part)  # Shuffle within the bin.
                out_index.append(part)
            seq_index = numpy.concatenate(out_index)
        elif self.seq_ordering.startswith("laplace"):
            assert get_seq_len
            tmp = self.seq_ordering.split(":")[1:]
            if len(tmp) == 0:
                bins = 2
            else:
                if tmp[0].startswith("."):  # starting with "." -> approx chunk size (num of seqs in one bin)
                    bins = max(num_seqs // int(tmp[0][1:]), 2)
                else:  # the number of bins
                    bins = int(tmp[0])
            if len(tmp) <= 1:
                nth = 1
            else:
                nth = int(tmp[1])
            rnd_seed = self._get_random_seed_for_epoch(epoch=epoch, num_epochs_fixed=nth)
            random_generator = numpy.random.RandomState(rnd_seed)
            seq_index = random_generator.permutation(num_seqs)  # type: Union[numpy.ndarray, List[int]]
            out_index = []
            for i in range(bins):
                if i == bins - 1:
                    part = seq_index[i * len(seq_index) // bins :].tolist()
                else:
                    part = seq_index[i * len(seq_index) // bins : (i + 1) * len(seq_index) // bins].tolist()
                part.sort(key=get_seq_len, reverse=(i % 2 == 1))
                out_index += part
            seq_index = out_index
        else:
            assert False, "invalid batching specified: " + self.seq_ordering

        if self.unique_seq_tags:
            # Note: This is as generic as possible, but requires that get_all_tags is implemented.
            all_seq_tags = self.get_all_tags()
            used_seq_tags = set()
            seq_index = [
                i for i in seq_index if (all_seq_tags[i] not in used_seq_tags, used_seq_tags.add(all_seq_tags[i]))[0]
            ]
        if partition_epoch > 1:
            seq_index = self._apply_partition_epoch(seq_index, partition_epoch, epoch)
        if repeat_epoch > 1:
            seq_index = list(seq_index) * repeat_epoch
        if self.seq_tags_filter is not None:
            # Note: This is as generic as possible, but requires that get_all_tags is implemented.
            assert len(seq_index)
            all_seq_tags = self.get_all_tags()
            assert len(all_seq_tags) == num_seqs == self.get_total_num_seqs(), "%r vs %r vs %r" % (
                len(all_seq_tags),
                num_seqs,
                self.get_total_num_seqs(),
            )
            old_seq_index = seq_index
            seq_index = [i for i in seq_index if all_seq_tags[i] in self.seq_tags_filter]
            assert (
                seq_index
            ), "%s: empty after applying seq_list_filter_file. Example filter tags: %r, used tags: %r" % (
                self,
                sorted(self.seq_tags_filter)[:3],
                [all_seq_tags[i] for i in old_seq_index[:3]],
            )
        return seq_index

    @classmethod
    def _apply_partition_epoch(cls, seq_index, partition_epoch, epoch):
        """
        :param typing.Sequence[int] seq_index: full list of ordered sequence indices
        :param int partition_epoch: number of partitions seq_index should be split into
        :param int|None epoch: current epoch
        :return: partition of seq_index for current epoch
        :rtype: typing.Sequence[int]
        """
        num_seqs = len(seq_index)
        current_partition = ((epoch or 1) - 1) % partition_epoch
        seqs_per_epoch = num_seqs // partition_epoch
        partition_sizes = [seqs_per_epoch + 1] * (num_seqs % partition_epoch) + [seqs_per_epoch] * (
            partition_epoch - num_seqs % partition_epoch
        )
        assert sum(partition_sizes) == num_seqs and len(partition_sizes) == partition_epoch
        partitions = functools.reduce(lambda a, x: a + [a[-1] + x], partition_sizes, [0])  # cumulative sum
        assert len(partitions) == partition_epoch + 1
        seq_index = seq_index[partitions[current_partition] : partitions[current_partition + 1]]
        assert len(seq_index) == partition_sizes[current_partition]

        return seq_index

    def _get_random_seed_for_epoch(self, epoch, num_epochs_fixed=1):
        """
        :param int|None epoch:
        :param int num_epochs_fixed: keep random seed fixed for n subsequent epochs
        :rtype: int
        """
        if self.fixed_random_seed is not None:
            return self.fixed_random_seed
        partition_epoch = self.partition_epoch or 1
        seed = epoch or 1
        if partition_epoch > 1:
            seed = (seed - 1) // partition_epoch + 1  # taking partitions requires constant seed during full epoch
        if num_epochs_fixed > 1:
            seed = (seed - 1) // num_epochs_fixed + 1
        return seed + self.random_seed_offset

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """
        :type epoch: int|None
        :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
        :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order. Only possible
          if the dataset has such indices (see self.have_corpus_seq_idx()).
        :rtype: bool
        :returns whether the order changed (True is always safe to return)

        This is called when we start a new epoch, or at initialization.
        Call this when you reset the seq list.
        """
        self.epoch = epoch
        self.rnd_seq_drop = Random(self._get_random_seed_for_epoch(epoch=epoch))
        return False

    def finish_epoch(self):
        """
        This would get called at the end of the epoch (currently optional only).
        After this, further calls to :func:`get_data` or :func:`load_seqs` are invalid,
        until a new call to :func:`init_seq_order` follows.
        """
        self.epoch = None

    def get_current_seq_order(self):
        """
        :return: many datasets use self.get_seq_order_for_epoch. this function would return the current seq order
          for the current epoch, after self.init_seq_order was called.
          Not all datasets implement this.
        :rtype: typing.Sequence[int]
        """
        raise OptionalNotImplementedError

    def initialize(self):
        """
        Does the main initialization before it can be used.
        This needs to be called before self.load_seqs() can be used.
        """
        assert self.num_outputs
        self.init_seq_order()

    def get_times(self, sorted_seq_idx):
        """
        :param int sorted_seq_idx:
        """
        raise OptionalNotImplementedError

    def get_data(self, seq_idx, key) -> numpy.ndarray:
        """
        :param int seq_idx: sorted seq idx
        :param str key: data-key, e.g. "data" or "classes"
        :return: features or targets: format 2d (time,feature) (float)
        """
        # Fallback implementation for old-style subclasses.
        if key == "data":
            return self.get_input_data(seq_idx)
        else:
            return self.get_targets(key, seq_idx)

    def get_input_data(self, sorted_seq_idx):
        """
        :type sorted_seq_idx: int
        :rtype: numpy.ndarray
        :returns features: format 2d (time,feature) (float)
        """
        raise NotImplementedError

    def get_targets(self, target, sorted_seq_idx):
        """
        :param str target: data key
        :type sorted_seq_idx: int
        :rtype: numpy.ndarray
        :returns targets: format 1d (time) (int: idx of output-feature)
        """
        # For new-style subclasses, which just provide get_data.
        return self.get_data(sorted_seq_idx, target)

    def get_tag(self, sorted_seq_idx):
        """
        :param int sorted_seq_idx:
        :rtype: str
        """
        return "seq-%i" % sorted_seq_idx

    def get_all_tags(self):
        """
        :return: list of all seq tags, of the whole dataset, without partition epoch.
          Note that this is not possible with all datasets.
        :rtype: list[str]
        """
        old_partition_epoch = self.partition_epoch
        try:
            all_tags = [None] * self.num_seqs  # type: typing.List[typing.Union[None,str]]
            for seq_idx in range(self.num_seqs):
                all_tags[seq_idx] = self.get_tag(seq_idx)
            return all_tags
        finally:
            self.partition_epoch = old_partition_epoch

    def get_total_num_seqs(self) -> int:
        """
        :return: total number of seqs, without partition epoch.
          Should be the same as len(self.get_all_tags()).
          Note that this is not possible with all datasets.
        """
        if self.partition_epoch == 1:
            # Note: self.num_seqs might not always be set, or even be correct...
            return self.num_seqs
        raise NotImplementedError("%s: get_total_num_seqs with partition epoch %i" % (self, self.partition_epoch))

    def have_corpus_seq_idx(self):
        """
        :rtype: bool
        :return: whether you can call self.get_corpus_seq_idx()
        """
        return False

    def get_corpus_seq_idx(self, seq_idx):
        """
        :param int seq_idx: sorted sequence index from the current epoch, depending on seq_ordering
        :return: the sequence index as-is in the original corpus (as if you would have sorting="default").
          only defined if self.have_corpus_seq_idx()
        :rtype: int
        """
        if self.seq_ordering == "default":
            return seq_idx
        assert self.have_corpus_seq_idx()
        raise NotImplemented

    def have_get_corpus_seq(self) -> bool:
        """
        :return: whether you can call :func:`get_corpus_seq`
        """
        return False

    def get_corpus_seq(self, corpus_seq_idx: int) -> DatasetSeq:
        """
        This function allows random access directly into the corpus.
        Only implement this if such random access is possible in a reasonable efficient way.
        This allows to write map-style wrapper datasets around such RETURNN datasets.

        :param corpus_seq_idx: corresponds to output of :func:`get_corpus_seq_idx`
        :return: data
        """
        raise OptionalNotImplementedError

    @classmethod
    def generic_complete_frac(cls, seq_idx, num_seqs):
        """
        :param int seq_idx: idx
        :param int|None num_seqs: None if not available
        :return: Returns a fraction (float in [0,1], always > 0) of how far we have advanced
          for this seq in the dataset.
          This does not have to be exact. This is only for the user.
        """
        if num_seqs:
            return min(float(seq_idx + 1) / num_seqs, 1.0)
        else:
            # We don't know. So:
            # Some monotonic increasing function in [0,1] which never reaches 1.
            import math

            return max(1.0e-10, 1.0 - math.exp(-seq_idx * 1000))

    def get_complete_frac(self, seq_idx):
        """
        :param int seq_idx:
        :return: Returns a fraction (float in [0,1], always > 0) of how far we have advanced
          for this seq in the dataset.
          This does not have to be exact. This is only for the user.
        :rtype: float
        """
        # noinspection PyBroadException
        try:
            num_seqs = self.num_seqs
        except Exception:  # num_seqs not always available
            # noinspection PyBroadException
            try:
                num_seqs = self.estimated_num_seqs
            except Exception:  # also not always available
                num_seqs = None  # ignore
        return self.generic_complete_frac(seq_idx, num_seqs)

    @property
    def num_seqs(self) -> int:
        """
        :return: num seqs for current epoch
        """
        raise NotImplementedError

    @property
    def estimated_num_seqs(self):
        """
        :return: estimated num seqs. does not have to be exact
        :rtype: int|None
        """
        # noinspection PyBroadException
        try:
            return self.num_seqs
        except Exception:  # might not be available
            pass
        if self._estimated_num_seqs is not None:
            return self._estimated_num_seqs
        return None

    def get_data_keys(self):
        """
        :return: all available data keys (for get_data and all other functions)
        :rtype: list[str]
        """
        return ["data"] + self.get_target_list()

    def get_target_list(self):
        """
        :return: subset of :func:`get_data_keys`. target keys are usually not available during inference
        :rtype: list[str]
        """
        return ["classes"]

    def get_data_dim(self, key):
        """
        :param str key: e.g. "data" or "classes"
        :return: number of classes, no matter if sparse or not
        :rtype: int
        """
        if key in self.num_outputs:
            # num_outputs should have the correct dimension, even for key "data" with self.window > 1.
            return self.num_outputs[key][0]
        return 1  # unknown

    def get_data_dtype(self, key):
        """
        :param str key: e.g. "data" or "classes"
        :return: dtype as str, e.g. "int32" or "float32"
        :rtype: str
        """
        if self.is_data_sparse(key):
            return "int32"
        return "float32"

    def is_data_sparse(self, key):
        """
        :param str key: e.g. "data" or "classes"
        :return: whether the data is sparse
        :rtype: bool
        """
        # Note: We cannot call get_data_dtype, as we would maybe result in infinite recursion.
        if key in self.num_outputs:
            return self.num_outputs[key][1] <= 1
        if key == "data":
            return False
        return True

    def get_data_shape(self, key: str) -> List[int]:
        """
        :returns get_data(*, key).shape[1:], i.e. num-frames excluded
        """
        if key in self.num_outputs:
            if self.num_outputs[key][1] <= 1:
                return []
            res_shape = [None] * (self.num_outputs[key][1] - 1)  # type: typing.List[typing.Union[None,int]]
            if not self.is_data_sparse(key):
                res_shape[-1] = self.get_data_dim(key)
            return res_shape
        if self.is_data_sparse(key):
            return []
        return [self.get_data_dim(key)]

    def have_seqs(self) -> bool:
        """
        :return: whether num_seqs > 0
        """
        try:
            total_num_seqs = self.get_total_num_seqs()
            return total_num_seqs > 0
        except NotImplementedError:
            pass
        return self.is_less_than_num_seqs(0)

    def len_info(self):
        """
        :rtype: str
        :returns a string to present the user as information about our len.
        Depending on our implementation, we can give some more or some less information.
        """
        return ", ".join(
            [
                self.__class__.__name__,
                "sequences: %s" % try_run(lambda: self.num_seqs, default="unknown"),
                "frames: %s" % try_run(self.get_num_timesteps, default="unknown"),
            ]
        )

    def is_less_than_num_seqs(self, n):
        """
        :type n: int
        :rtype: bool
        :returns whether n < num_seqs. In case num_seqs is not known in advance, it will wait
        until it knows that n is behind the end or that we have the seq.
        """
        # We keep this dynamic so that other implementations which don't know the num of seqs
        # in advance can handle this somehow.
        return n < self.num_seqs

    def can_serialize_data(self, key):
        """
        :param str key: e.g. "classes"
        :rtype: bool
        """
        return key in self.labels

    def serialize_data(self, key, data):
        """
        In case you have a :class:`Vocabulary`, just use :func:`Vocabulary.get_seq_labels`.

        :param str key: e.g. "classes". self.labels[key] should be set
        :param numpy.ndarray data: 0D or 1D
        :rtype: str
        """
        vocab = Vocabulary.create_vocab_from_labels(self.labels[key])
        if data.ndim == 0:
            return vocab.labels[data]
        return vocab.get_seq_labels(data)

    def sample(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: bool
        """
        if seq_idx in self.weights:
            weight = self.weights[seq_idx]
            return weight[0] >= weight[1]
        return True


class DatasetSeq:
    """
    Encapsulates all data for one sequence.
    """

    def __init__(self, seq_idx, features, targets=None, seq_tag=None):
        """
        :param int seq_idx: sorted seq idx in the Dataset
        :param numpy.ndarray|dict[str,numpy.ndarray] features: format 2d (time,feature) (float)
        :param dict[str,numpy.ndarray]|numpy.ndarray|None targets: name -> format 1d (time) (idx of output-feature)
        :param str seq_tag: sequence name / tag
        """
        assert isinstance(seq_idx, (int, numpy.integer))
        self.seq_idx = int(seq_idx)
        self.seq_tag = seq_tag or ("seq-%i" % seq_idx)
        if not isinstance(features, dict):
            assert isinstance(features, numpy.ndarray)
            features = {"data": features}
            if targets is None:
                targets = {}
            if isinstance(targets, numpy.ndarray):  # old format
                targets = {"classes": targets}
            assert isinstance(targets, dict)
            features.update(targets)
            targets = None
        assert isinstance(features, dict) and targets is None
        for v in features.values():
            assert isinstance(v, numpy.ndarray)
        self.features = features

    @property
    def num_frames(self):
        """
        :rtype: NumbersDict
        """
        d = {k: (v.shape[0] if v.ndim >= 1 else 1) for (k, v) in self.features.items()}
        return NumbersDict(d)

    def get_data(self, key):
        """
        :param str key:
        :rtype: numpy.ndarray
        """
        return self.features[key]

    def get_data_keys(self):
        """
        :rtype: set[str]
        """
        return self.features.keys()

    def __repr__(self):
        return "<DataCache seq_idx=%i>" % self.seq_idx


_dataset_classes = {}  # type: Dict[str,Type[Dataset]]


def get_dataset_class(name: Union[str, Type[Dataset]]) -> Optional[Type[Dataset]]:
    """
    :param str|type name:
    """
    if isinstance(name, type):
        assert issubclass(name, Dataset)
        return name

    if _dataset_classes:
        return _dataset_classes.get(name, None)

    from importlib import import_module

    # Only those modules which make sense to be loaded by the user,
    # because this function is only used for such cases.
    mod_names = ["hdf", "sprint", "generating", "numpy_dump", "meta", "lm", "map", "multi_proc"]
    for mod_name in mod_names:
        mod = import_module("returnn.datasets.%s" % mod_name)
        for name_, clazz in vars(mod).items():
            if name_ in _dataset_classes:  # prefer first
                continue
            if not isinstance(clazz, type) or not issubclass(clazz, Dataset):
                continue
            _dataset_classes[name_] = clazz

    return _dataset_classes.get(name, None)


def init_dataset(kwargs, extra_kwargs=None, default_kwargs=None):
    """
    :param dict[str]|str|(()->dict[str])|Dataset kwargs:
    :param dict[str]|None extra_kwargs:
    :param dict[str]|None default_kwargs:
    :rtype: Dataset
    """
    assert kwargs
    if isinstance(kwargs, Dataset):
        data = kwargs
        data.initialize()
        return data
    if callable(kwargs):
        return init_dataset(kwargs(), extra_kwargs=extra_kwargs, default_kwargs=default_kwargs)
    if isinstance(kwargs, str):
        if kwargs.startswith("{"):
            kwargs = eval(kwargs)
        else:
            assert ValueError("Defining datasets via string is no longer allowed")
    assert isinstance(kwargs, dict)
    kwargs = kwargs.copy()
    assert "class" in kwargs
    clazz_name = kwargs.pop("class")
    clazz = get_dataset_class(clazz_name)
    if not clazz:
        raise Exception("Dataset class %r not found" % clazz_name)
    if default_kwargs:
        for key, value in default_kwargs.items():
            kwargs.setdefault(key, value)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    obj = clazz(**kwargs)
    assert isinstance(obj, Dataset)
    obj.initialize()
    return obj


def convert_data_dims(data_dims, leave_dict_as_is=False):
    """
    This converts what we called num_outputs originally,
    from the various formats which were allowed in the past
    (just an int, or dict[str,int]) into the format which we currently expect.
    In all cases, the output will be a new copy of the dict.

    :param int|dict[str,int|(int,int)|dict] data_dims: what we called num_outputs originally
    :param bool leave_dict_as_is:
    :rtype: dict[str,(int,int)|dict]
    :returns dict data-key -> (data-dimension, len(shape) (1 ==> sparse))
     (or potentially data-key -> dict, if leave_dict_as_is is True; for TensorFlow)
    """
    if isinstance(data_dims, int):
        data_dims = {"classes": data_dims}
    assert isinstance(data_dims, dict)
    data_dims = data_dims.copy()
    for k, v in list(data_dims.items()):
        if isinstance(v, int):
            v = (v, 2 if k == "data" else 1)
            data_dims[k] = v
        if isinstance(v, dict) and leave_dict_as_is:
            continue
        assert isinstance(v, (tuple, list))
        data_dims[k] = tuple(v)
        assert len(v) == 2
        assert isinstance(v[0], int)
        assert isinstance(v[1], int)
        assert 1 <= v[1]
    return data_dims
