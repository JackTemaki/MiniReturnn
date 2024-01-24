"""
Provides :class:`HDFDataset`.
"""

from __future__ import annotations
import typing
import gc
import h5py
import numpy
from .cached import CachedDataset
from .basic import DatasetSeq
from returnn.log import log
from .util.hdf import HDF_SEQ_LENGTHS_KEY, HDF_INPUT_PATT_SIZE_KEY, HDF_TIMES_KEY
from .util.hdf import SimpleHDFWriter, HDFDatasetWriter  # noqa, classes were moved

# Common attribute names for HDF dataset, which should be used in order to be proceed with HDFDataset class.


class HDFDataset(CachedDataset):
    """
    Dataset based on HDF files.
    This was the main original dataset format of RETURNN.
    """

    def __init__(self, files=None, use_cache_manager=False, **kwargs):
        """
        :param None|list[str] files:
        :param bool use_cache_manager: uses :func:`Util.cf` for files
        """
        super(HDFDataset, self).__init__(**kwargs)
        assert (
            self.partition_epoch == 1 or self.cache_byte_size_total_limit == 0
        ), "To use partition_epoch in HDFDatasets, disable caching by setting cache_byte_size=0"
        if files is not None and not isinstance(files, list):
            raise TypeError("HDFDataset 'files' have to be defined as a list")
        self._use_cache_manager = use_cache_manager
        self.files = []  # type: typing.List[str]  # file names
        self.h5_files = []  # type: typing.List[h5py.File]
        # We cache the h5py.Dataset objects that are created each time when accessing a h5py.File,
        # e.g. via fin['inputs'],
        # as this access seems to have a significant overhead.
        # Speeds up going through a HDFDataset by up to factor 3
        # (tested with h5py 3.1.0).
        self.cached_h5_datasets = []  # type: typing.List[typing.Dict[str,h5py.Dataset]]
        self.file_start = [0]
        self.file_seq_start = []  # type: typing.List[numpy.ndarray]
        self.data_dtype = {}  # type: typing.Dict[str,str]
        self.data_sparse = {}  # type: typing.Dict[str,bool]
        self._num_codesteps = None  # type: typing.Optional[typing.List[int]]  # accumulated sequence length per target

        if files:
            for fn in files:
                self.add_file(fn)

    def __del__(self):
        for f in self.h5_files:
            # noinspection PyBroadException
            try:
                f.close()
            except Exception:  # e.g. at shutdown. but does not matter
                pass
        del self.h5_files[:]
        del self.file_seq_start[:]

    @staticmethod
    def _decode(s):
        """
        :param str|bytes|unicode s:
        :rtype: str
        """
        if not isinstance(s, str):  # bytes (Python 3)
            s = s.decode("utf-8")
        s = s.split("\0")[0]
        return s

    def add_file(self, filename):
        """
        Setups data:
          self.file_start
          self.file_seq_start
        Use load_seqs() to load the actual data.
        :type filename: str
        """
        if self._use_cache_manager:
            from returnn.util.basic import cf

            filename = cf(filename)
        fin = h5py.File(filename, "r")
        if "targets" in fin:
            self.labels = {
                k: [self._decode(item) for item in fin["targets/labels"][k][...].tolist()]
                for k in fin["targets/labels"]
            }
        if not self.labels and "labels" in fin:
            labels = [item.split("\0")[0] for item in fin["labels"][...].tolist()]  # type: typing.List[str]
            self.labels = {"classes": labels}
            assert len(self.labels["classes"]) == len(labels), (
                "expected " + str(len(self.labels["classes"])) + " got " + str(len(labels))
            )
        self.files.append(filename)
        self.h5_files.append(fin)
        self.cached_h5_datasets.append({})
        print("parsing file", filename, file=log.v5)
        if "times" in fin:
            if self.timestamps is None:
                self.timestamps = fin[HDF_TIMES_KEY][...]
            else:
                self.timestamps = numpy.concatenate([self.timestamps, fin[HDF_TIMES_KEY][...]], axis=0)
        prev_target_keys = None
        if len(self.files) >= 2:
            prev_target_keys = self.target_keys
        if "targets" in fin:
            self.target_keys = sorted(set(fin["targets/data"].keys()) | set(fin["targets/size"].attrs.keys()))
        else:
            self.target_keys = ["classes"]

        seq_lengths = fin[HDF_SEQ_LENGTHS_KEY][...]  # shape (num_seqs,num_target_keys + 1)
        num_input_keys = 1 if "inputs" in fin else 0
        if len(seq_lengths.shape) == 1:
            seq_lengths = numpy.array(
                zip(*[seq_lengths.tolist() for _ in range(num_input_keys + len(self.target_keys))])
            )
        assert seq_lengths.ndim == 2 and seq_lengths.shape[1] == num_input_keys + len(self.target_keys)

        if prev_target_keys is not None and prev_target_keys != self.target_keys:
            print(
                "Warning: %s: loaded prev files %s, which defined target keys %s. Now loaded %s and got target keys %s."
                % (self, self.files[:-1], prev_target_keys, filename, self.target_keys),
                file=log.v2,
            )
            # This can happen for multiple reasons. E.g. just different files. Or saved with different RETURNN versions.
            # We currently support this by removing all the new additional targets, which only works if the prev targets
            # were a subset (so the order in which you load the files matters).
            assert all([key in self.target_keys for key in prev_target_keys])  # check if subset
            # Filter out the relevant seq lengths
            seq_lengths = seq_lengths[:, [0] + [self.target_keys.index(key) + 1 for key in prev_target_keys]]
            assert seq_lengths.shape[1] == len(prev_target_keys) + 1
            self.target_keys = prev_target_keys

        seq_start = numpy.zeros((seq_lengths.shape[0] + 1, seq_lengths.shape[1]), dtype="int64")
        numpy.cumsum(seq_lengths, axis=0, dtype="int64", out=seq_start[1:])

        self._num_timesteps += numpy.sum(seq_lengths[:, 0])
        if self._num_codesteps is None:
            self._num_codesteps = [0 for _ in range(num_input_keys, len(seq_lengths[0]))]
        for i in range(num_input_keys, len(seq_lengths[0])):
            self._num_codesteps[i - 1] += numpy.sum(seq_lengths[:, i])

        if not self._seq_start:
            self._seq_start = [numpy.zeros((seq_lengths.shape[1],), "int64")]

        # May be large, so better delete them early, we don't need them anymore.
        del seq_lengths

        self.file_seq_start.append(seq_start)
        nseqs = len(seq_start) - 1
        self._num_seqs += nseqs
        self.file_start.append(self.file_start[-1] + nseqs)

        if "inputs" in fin:
            assert "data" not in self.target_keys, "Cannot use 'data' key for both a target and 'inputs'."
            if len(fin["inputs"].shape) == 1:  # sparse
                num_inputs = [int(fin.attrs[HDF_INPUT_PATT_SIZE_KEY]), 1]
            else:
                num_inputs = [
                    int(fin["inputs"].shape[1]),
                    len(fin["inputs"].shape),
                ]  # fin.attrs[HDF_INPUT_PATT_SIZE_KEY]
        else:
            num_inputs = [0, 0]
        if self.num_inputs == 0:
            self.num_inputs = num_inputs[0]
        assert self.num_inputs == num_inputs[0], "wrong input dimension in file %s (expected %s got %s)" % (
            filename,
            self.num_inputs,
            num_inputs[0],
        )
        num_outputs = {}
        if "targets/size" in fin:
            for k in self.target_keys:
                if numpy.isscalar(fin["targets/size"].attrs[k]):
                    num_outputs[k] = (int(fin["targets/size"].attrs[k]), len(fin["targets/data"][k].shape))
                else:  # hdf_dump will give directly as tuple
                    assert fin["targets/size"].attrs[k].shape == (2,)
                    num_outputs[k] = tuple([int(v) for v in fin["targets/size"].attrs[k]])
        if "inputs" in fin:
            num_outputs["data"] = num_inputs
        if not self.num_outputs:
            self.num_outputs = num_outputs
        assert self.num_outputs == num_outputs, "wrong dimensions in file %s (expected %s got %s)" % (
            filename,
            self.num_outputs,
            num_outputs,
        )

        if "targets" in fin:
            for name in self.target_keys:
                self.data_dtype[str(name)] = str(fin["targets/data"][name].dtype)
                self.targets[str(name)] = None
                if str(name) not in self.num_outputs:
                    ndim = len(fin["targets/data"][name].shape)
                    dim = 1 if ndim == 1 else fin["targets/data"][name].shape[-1]
                    self.num_outputs[str(name)] = (dim, ndim)
        if "inputs" in fin:
            self.data_dtype["data"] = str(fin["inputs"].dtype)
        assert num_input_keys + len(self.target_keys) == len(self.file_seq_start[0][0])

    def _load_seqs(self, start, end):
        """
        Load data sequences.
        As a side effect, will modify / fill-up:
          self.alloc_intervals
          self.targets
          self.chars

        :param int start: start sorted seq idx
        :param int end: end sorted seq idx
        """
        assert start < self.num_seqs
        assert end <= self.num_seqs
        if self.cache_byte_size_total_limit == 0:
            # Just don't use the alloc intervals, or any of the other logic. Just load it on the fly when requested.
            return
        selection = self.insert_alloc_interval(start, end)
        assert len(selection) <= end - start, (
            "DEBUG: more sequences requested (" + str(len(selection)) + ") as required (" + str(end - start) + ")"
        )
        self.preload_set |= set(range(start, end)) - set(selection)
        file_info = [[] for _ in range(len(self.files))]  # type: typing.List[typing.List[typing.Tuple[int,int]]]
        # file_info[i] is (sorted seq idx from selection, real seq idx)
        for idc in selection:
            if self.sample(idc):
                ids = self._seq_index[idc]
                file_info[self._get_file_index(ids)].append((idc, ids))
            else:
                self.preload_set.add(idc)
        for i in range(len(self.files)):
            if len(file_info[i]) == 0:
                continue
            if start == 0 or self.cache_byte_size_total_limit > 0:  # suppress with disabled cache
                print(
                    "loading file %d/%d (seq range %i-%i)" % (i + 1, len(self.files), start, end),
                    self.files[i],
                    file=log.v4,
                )
            fin = self.h5_files[i]
            inputs = fin["inputs"] if self.num_inputs > 0 else None
            targets = None
            if "targets" in fin:
                targets = {k: fin["targets/data/" + k] for k in fin["targets/data"]}
            for idc, ids in file_info[i]:
                s = ids - self.file_start[i]
                p = self.file_seq_start[i][s]
                q = self.file_seq_start[i][s + 1]
                if "targets" in fin:
                    for k in fin["targets/data"]:
                        if self.targets[k] is None:
                            self.targets[k] = (
                                numpy.zeros(
                                    (self._num_codesteps[self.target_keys.index(k)],) + targets[k].shape[1:],
                                    dtype=self.data_dtype[k],
                                )
                                - 1
                            )
                        ldx = self.target_keys.index(k) + 1
                        self.targets[k][
                            self.get_seq_start(idc)[ldx] : self.get_seq_start(idc)[ldx] + q[ldx] - p[ldx]
                        ] = targets[k][p[ldx] : q[ldx]]
                if inputs:
                    self._set_alloc_intervals_data(idc, data=inputs[p[0] : q[0]])
                self.preload_set.add(idc)
        gc.collect()

    def get_data(self, seq_idx, key):
        """
        :param int seq_idx:
        :param str key:
        :rtype: numpy.ndarray
        """
        if self.cache_byte_size_total_limit > 0:  # Use the cache?
            return super(HDFDataset, self).get_data(seq_idx, key)

        # Otherwise, directly read it from file now.
        real_seq_idx = self._seq_index[seq_idx]
        return self._get_data_by_real_seq_idx(real_seq_idx, key)

    def get_data_by_seq_tag(self, seq_tag, key):
        """
        :param str seq_tag:
        :param str key:
        :rtype: numpy.ndarray
        """
        if self.cache_byte_size_total_limit > 0:  # Use the cache?
            raise Exception("%s: get_data_by_seq_tag not supported with cache" % self)

        # Otherwise, directly read it from file now.
        self._update_tag_idx()
        real_seq_idx = self._tag_idx[seq_tag]
        return self._get_data_by_real_seq_idx(real_seq_idx, key)

    def _get_data_by_real_seq_idx(self, real_seq_idx, key):
        """
        :param int real_seq_idx:
        :param str key:
        :rtype: numpy.ndarray
        """
        file_idx = self._get_file_index(real_seq_idx)
        fin = self.h5_files[file_idx]

        real_file_seq_idx = real_seq_idx - self.file_start[file_idx]
        start_pos = self.file_seq_start[file_idx][real_file_seq_idx]
        end_pos = self.file_seq_start[file_idx][real_file_seq_idx + 1]

        if key == "data" and self.num_inputs > 0:
            if "inputs" not in self.cached_h5_datasets[file_idx]:
                assert "inputs" in fin
                self.cached_h5_datasets[file_idx]["inputs"] = fin[
                    "inputs"
                ]  # cached for efficiency, see comment in __init__()

            inputs = self.cached_h5_datasets[file_idx]["inputs"]
            data = inputs[start_pos[0] : end_pos[0]]

        else:
            if key not in self.cached_h5_datasets[file_idx]:
                assert "targets" in fin
                self.cached_h5_datasets[file_idx][key] = fin["targets/data/" + key]  # see comment in __init__()

            targets = self.cached_h5_datasets[file_idx][key]
            first_target_idx = 1 if self.num_inputs > 0 else 0  # self.num_inputs == 0 if no 'inputs' in HDF file
            ldx = first_target_idx + self.target_keys.index(key)
            data = targets[start_pos[ldx] : end_pos[ldx]]

        return data

    def get_input_data(self, sorted_seq_idx):
        """
        :param int sorted_seq_idx:
        :rtype: numpy.ndarray
        """
        if self.cache_byte_size_total_limit > 0:  # Use the cache?
            return super(HDFDataset, self).get_input_data(sorted_seq_idx)
        return self.get_data(sorted_seq_idx, "data")

    def get_targets(self, target, sorted_seq_idx):
        """
        :param str target:
        :param int sorted_seq_idx:
        :rtype: numpy.ndarray
        """
        if self.cache_byte_size_total_limit > 0:  # Use the cache?
            return super(HDFDataset, self).get_targets(target, sorted_seq_idx)
        return self.get_data(sorted_seq_idx, target)

    def get_estimated_seq_length(self, seq_idx):
        """
        :param int seq_idx: for current epoch, not the corpus seq idx
        :rtype: int
        :returns sequence length of "data", used for sequence sorting
        """
        real_seq_idx = self._seq_index[self._index_map[seq_idx]]
        return int(self._get_seq_length_by_real_idx(real_seq_idx)[0])

    def _get_seq_length_by_real_idx(self, real_seq_idx):
        """
        :param int real_seq_idx:
        :returns length of the sequence with index 'real_seq_idx'. see get_seq_length_nd
        :rtype: numpy.ndarray
        """
        file_idx = self._get_file_index(real_seq_idx)
        real_file_seq_idx = real_seq_idx - self.file_start[file_idx]

        start_pos = self.file_seq_start[file_idx][real_file_seq_idx]
        end_pos = self.file_seq_start[file_idx][real_file_seq_idx + 1]

        return end_pos - start_pos

    def _get_tag_by_real_idx(self, real_seq_idx):
        file_idx = self._get_file_index(real_seq_idx)
        real_file_seq_idx = real_seq_idx - self.file_start[file_idx]

        if "#seqTags" not in self.cached_h5_datasets[file_idx]:
            self.cached_h5_datasets[file_idx]["#seqTags"] = self.h5_files[file_idx]["seqTags"]

        s = self.cached_h5_datasets[file_idx]["#seqTags"][real_file_seq_idx]
        s = self._decode(s)
        return s

    def get_tag(self, sorted_seq_idx):
        """
        :param int sorted_seq_idx:
        :rtype: str
        """
        ids = self._seq_index[self._index_map[sorted_seq_idx]]
        return self._get_tag_by_real_idx(ids)

    def have_get_corpus_seq(self) -> bool:
        """
        :return: whether this dataset supports :func:`get_corpus_seq`
        """
        return True

    def get_corpus_seq(self, corpus_seq_idx: int) -> DatasetSeq:
        """
        :param int corpus_seq_idx: corpus seq idx
        :return: the seq with the given corpus seq idx
        :rtype: DatasetSeq
        """
        data = {}
        for key in self.get_data_keys():
            data[key] = self._get_data_by_real_seq_idx(corpus_seq_idx, key)
        return DatasetSeq(seq_idx=corpus_seq_idx, features=data, seq_tag=self._get_tag_by_real_idx(corpus_seq_idx))

    def get_all_tags(self):
        """
        :rtype: list[str]
        """
        tags = []
        for h5_file in self.h5_files:
            tags += h5_file["seqTags"][...].tolist()
        return list(map(self._decode, tags))

    def get_total_num_seqs(self):
        """
        :rtype: int
        """
        return self._num_seqs

    def is_data_sparse(self, key):
        """
        :param str key:
        :rtype: bool
        """
        if "int" in self.get_data_dtype(key):
            if key in self.num_outputs:
                return self.num_outputs[key][1] <= 1
        return False

    def get_data_dtype(self, key):
        """
        :param str key:
        :rtype: str
        """
        return self.data_dtype[key]

    def len_info(self):
        """
        :rtype: str
        """
        return ", ".join(["HDF dataset", "sequences: %i" % self.num_seqs, "frames: %i" % self.get_num_timesteps()])

    def _get_file_index(self, real_seq_idx):
        file_index = 0
        while file_index < len(self.file_start) - 1 and real_seq_idx >= self.file_start[file_index + 1]:
            file_index += 1
        return file_index
