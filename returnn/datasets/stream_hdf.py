from __future__ import annotations

import collections

import h5py
import numpy

from returnn.datasets.basic import DatasetSeq
from returnn.datasets.cached2 import CachedDataset2


class StreamParser(object):
    """
    Stream parser.
    """

    def __init__(self, seq_names, stream):
        self.seq_names = seq_names
        self.stream = stream

        self.num_features = None
        self.feature_type = None  # 1 for sparse, 2 for dense
        self.dtype = None

    def get_data(self, seq_name):
        """
        :param str seq_name:
        :rtype: numpy.ndarray
        """
        raise NotImplementedError()

    def get_seq_length(self, seq_name):
        """
        :param str seq_name:
        :rtype: int
        """
        raise NotImplementedError()

    def get_dtype(self):
        """
        :rtype: str
        """
        assert isinstance(self.dtype, str)
        return self.dtype


class FeatureSequenceStreamParser(StreamParser):
    """
    Feature sequence stream parser.
    """

    def __init__(self, *args, **kwargs):
        super(FeatureSequenceStreamParser, self).__init__(*args, **kwargs)

        for s in self.seq_names:
            seq_data = self.stream["data"][s]
            assert len(seq_data.shape) == 2

            if self.num_features is None:
                self.num_features = seq_data.shape[1]
            if self.dtype is None:
                self.dtype = str(seq_data.dtype)

            assert seq_data.shape[1] == self.num_features
            assert str(seq_data.dtype) == self.dtype

        self.feature_type = 2

    def get_data(self, seq_name):
        """
        :param str seq_name:
        :rtype: numpy.ndarray
        """
        return self.stream["data"][seq_name][...]

    def get_seq_length(self, seq_name):
        """
        :param str seq_name:
        :rtype: int
        """
        return self.stream["data"][seq_name].shape[0]


class SparseStreamParser(StreamParser):
    """
    Sparse stream parser.
    """

    def __init__(self, *args, **kwargs):
        super(SparseStreamParser, self).__init__(*args, **kwargs)

        for s in self.seq_names:
            seq_data = self.stream["data"][s]
            assert len(seq_data.shape) == 1

            if self.dtype is None:
                self.dtype = str(seq_data.dtype)
            assert str(seq_data.dtype) == self.dtype

        self.num_features = self.stream["feature_names"].shape[0]
        self.feature_type = 1

    def get_data(self, seq_name):
        """
        :param str seq_name:
        :rtype: numpy.ndarray
        """
        return self.stream["data"][seq_name][:]

    def get_seq_length(self, seq_name):
        """
        :param str seq_name:
        :rtype: int
        """
        return self.stream["data"][seq_name].shape[0]


class SegmentAlignmentStreamParser(StreamParser):
    """
    Segment alignment stream parser.
    """

    def __init__(self, *args, **kwargs):
        super(SegmentAlignmentStreamParser, self).__init__(*args, **kwargs)

        for s in self.seq_names:
            seq_data = self.stream["data"][s]

            if self.dtype is None:
                self.dtype = str(seq_data.dtype)
            assert str(seq_data.dtype) == self.dtype

            assert len(seq_data.shape) == 2
            assert seq_data.shape[1] == 2

        self.num_features = self.stream["feature_names"].shape[0]
        self.feature_type = 1

    def get_data(self, seq_name):
        """
        :param str seq_name:
        :return: flatted two-dimensional data where the 2nd dimension is 2 [class, segment end]
        :rtype: numpy.ndarray
        """
        length = self.get_seq_length(seq_name) // 2
        segments = self.stream["data"][seq_name][:]

        alignment = numpy.zeros((length, 2), dtype=self.dtype)
        num_segments = segments.shape[0]
        seg_end = 0
        for i in range(num_segments):
            next_seg_end = seg_end + segments[i, 1]
            alignment[seg_end:next_seg_end, 0] = segments[i, 0]  # set class
            alignment[next_seg_end - 1, 1] = 1  # mark segment end
            seg_end = next_seg_end

        alignment = alignment.reshape((-1,))
        return alignment

    def get_seq_length(self, seq_name):
        """
        :param str seq_name:
        :rtype: int
        """
        return 2 * sum(self.stream["data"][seq_name][:, 1])


class StreamHDFDataset(CachedDataset2):
    """
    Another separate dataset which uses HDF files to store the data.
    """

    parsers = {
        "feature_sequence": FeatureSequenceStreamParser,
        "sparse": SparseStreamParser,
        "segment_alignment": SegmentAlignmentStreamParser,
    }

    def __init__(self, input_stream_name, files=None, **kwargs):
        """
        :param str input_stream_name:
        :param None|list[str] files:
        """
        super(StreamHDFDataset, self).__init__(**kwargs)

        self.input_stream_name = input_stream_name

        self.files = []
        self.h5_files = []
        self.all_seq_names = []
        self.seq_name_to_idx = {}
        self.file_indices = []
        self.seq_order = []
        self.all_parsers = collections.defaultdict(list)

        if files:
            for fn in files:
                self.add_file(fn)

    def add_file(self, path):
        """
        :param str path:
        """
        self.files.append(path)
        self.h5_files.append(h5py.File(path))

        cur_file = self.h5_files[-1]

        assert {"seq_names", "streams"}.issubset(set(cur_file.keys())), (
            "%s does not contain all required datasets/groups" % path
        )

        seqs = list(cur_file["seq_names"])
        norm_seqs = [self._normalize_seq_name(s) for s in seqs]

        prev_no_seqs = len(self.all_seq_names)
        seqs_in_this_file = len(seqs)
        self.seq_name_to_idx.update(zip(seqs, range(prev_no_seqs, prev_no_seqs + seqs_in_this_file + 1)))

        self.all_seq_names.extend(seqs)
        self.file_indices.extend([len(self.files) - 1] * len(seqs))

        all_streams = set(cur_file["streams"].keys())
        assert self.input_stream_name in all_streams, "%s does not contain the input stream %s" % (
            path,
            self.input_stream_name,
        )

        parsers = {
            name: StreamHDFDataset.parsers[stream.attrs["parser"]](norm_seqs, stream)
            for name, stream in cur_file["streams"].items()
        }
        for k, v in parsers.items():
            self.all_parsers[k].append(v)

        if len(self.files) == 1:
            self.num_outputs = {name: [parser.num_features, parser.feature_type] for name, parser in parsers.items()}
            self.num_inputs = self.num_outputs[self.input_stream_name][0]
        else:
            num_features = [(name, self.num_outputs[name][0], parser.num_features) for name, parser in parsers.items()]
            assert all([nf[1] == nf[2] for nf in num_features]), "\n".join(
                [
                    "Number of features does not match for parser %s: %d (config) vs. %d (hdf-file)" % nf
                    for nf in num_features
                    if nf[1] != nf[2]
                ]
            )

    def initialize(self):
        """
        Initialization.
        """
        total_seqs = len(self.all_seq_names)
        self._num_seqs = total_seqs
        self._estimated_num_seqs = total_seqs

        super(StreamHDFDataset, self).initialize()

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """
        :type epoch: int|None
        :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
        :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order.
        """
        super(StreamHDFDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

        if seq_order is not None:
            self.seq_order = seq_order
        elif seq_list is not None:
            self.seq_order = [self.seq_name_to_idx[s] for s in seq_list]
        else:
            epoch = epoch or 1
            self.seq_order = self.get_seq_order_for_epoch(epoch, len(self.all_seq_names), self._get_seq_length)

    def _get_seq_length(self, orig_seq_idx):
        """
        :type orig_seq_idx: int
        :rtype int
        """
        parser = self.all_parsers[self.input_stream_name][self.file_indices[orig_seq_idx]]
        return parser.get_seq_length(self._normalize_seq_name(self.all_seq_names[orig_seq_idx]))

    def _collect_single_seq(self, seq_idx):
        """
        :type seq_idx: int
        :rtype: DatasetSeq
        """
        if seq_idx >= len(self.seq_order):
            return None

        real_seq_index = self.seq_order[seq_idx]
        file_index = self.file_indices[real_seq_index]
        seq_name = self.all_seq_names[real_seq_index]
        norm_seq_name = self._normalize_seq_name(seq_name)
        targets = {name: parsers[file_index].get_data(norm_seq_name) for name, parsers in self.all_parsers.items()}
        features = targets[self.input_stream_name]
        return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_name, features=features, targets=targets)

    def get_data_dtype(self, key):
        """
        :param str key: e.g. "data"
        :rtype: str
        """
        if key == "data":
            return self.get_data_dtype(self.input_stream_name)
        return self.all_parsers[key][0].get_dtype()

    @staticmethod
    def _normalize_seq_name(name):
        """
        HDF Datasets cannot contain '/' in their name (this would create subgroups), we do not
        want this and thus replace it with '\' when asking for data from the parsers
        :type name: string
        :rtype: string
        """
        return name.replace("/", "\\")
