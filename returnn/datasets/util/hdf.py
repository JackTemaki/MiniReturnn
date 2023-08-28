from __future__ import annotations

import typing

import h5py
import numpy

from returnn.log import log


HDF_SEQ_LENGTHS_KEY = "seqLengths"
HDF_INPUT_PATT_SIZE_KEY = "inputPattSize"
HDF_TIMES_KEY = "times"


class SimpleHDFWriter:
    """
    Intended for a simple interface, to dump data on-the-fly into a HDF file,
    which can be read later by :class:`HDFDataset`.

    Note that we dump to a temp file first, and only at :func:`close` we move it over to the real destination.
    """

    def __init__(self, filename, dim, labels=None, ndim=None, extra_type=None, swmr=False, extend_existing_file=False):
        """
        :param str filename: Create file, truncate if exists
        :param int|None dim:
        :param int ndim: counted without batch
        :param list[str]|None labels:
        :param dict[str,(int,int,str)]|None extra_type: key -> (dim,ndim,dtype)
        :param bool swmr: see https://docs.h5py.org/en/stable/swmr.html
        :param bool extend_existing_file: True also means we expect that it exists
        """
        from returnn.util.basic import hdf5_strings, unicode
        import tempfile
        import os
        import shutil

        if ndim is None:
            if dim is None:
                ndim = 1
            else:
                ndim = 2
        self.dim = dim
        self.ndim = ndim
        self.labels = labels
        if labels:
            assert len(labels) == dim
        self.filename = filename
        tmp_fd, self.tmp_filename = tempfile.mkstemp(suffix=".hdf")
        os.close(tmp_fd)
        self.extend_existing_file = extend_existing_file
        if extend_existing_file:
            assert os.path.exists(self.filename)
            shutil.copyfile(self.filename, self.tmp_filename)
        else:
            # By default, we should not override existing data.
            assert not os.path.exists(self.filename)
        self._file = h5py.File(
            self.tmp_filename, "r+" if extend_existing_file else "w", libver="latest" if swmr else None
        )

        if not extend_existing_file:
            self._file.attrs["numTimesteps"] = 0  # we will increment this on-the-fly
            self._file.attrs["inputPattSize"] = dim or 1
            self._file.attrs["numDims"] = 1  # ignored?
            self._file.attrs["numLabels"] = dim or 1
            self._file.attrs["numSeqs"] = 0  # we will increment this on-the-fly
            if labels:
                hdf5_strings(self._file, "labels", labels)
            else:
                self._file.create_dataset("labels", (0,), dtype="S5")  # dtype string length does not matter

        self._datasets = {}  # type: typing.Dict[str, h5py.Dataset]  # key -> data
        # seq_length idx represents (seq_idx,data_key_idx),
        # where data_key_idx == 0 is for the main input data,
        # and otherwise data_key_idx == 1 + sorted(self._prepared_extra).index(data_key).
        # data_key_idx must allow for 2 entries by default, as HDFDataset assumes 'classes' by default.
        if extend_existing_file:
            self._seq_lengths = self._file["seqLengths"]
        else:
            self._seq_lengths = self._file.create_dataset("seqLengths", (0, 2), dtype="i", maxshape=(None, None))
        # Note about strings in HDF: https://docs.h5py.org/en/stable/strings.html
        # Earlier we used S%i, i.e. fixed-sized strings, with the calculated max string length.
        if extend_existing_file:
            self._seq_tags = self._file["seqTags"]
        else:
            # noinspection PyUnresolvedReferences
            dt = h5py.special_dtype(vlen=unicode)
            self._seq_tags = self._file.create_dataset("seqTags", (0,), dtype=dt, maxshape=(None,))

        self._extra_num_time_steps = {}  # type: typing.Dict[str,int]  # key -> num-steps
        self._prepared_extra = set()
        if extra_type:
            self._prepare_extra(extra_type)

        if swmr:
            assert not self._file.swmr_mode  # this also checks whether the attribute exists (right version)
            self._file.swmr_mode = True
            # See comments in test_SimpleHDFWriter_swmr...
            raise NotImplementedError("SimpleHDFWriter SWMR is not really finished...")

    def __del__(self):
        if self._file:
            self._file.close()
            self._file = None

    def _prepare_extra(self, extra_type):
        """
        :param dict[str,(int,int,str)] extra_type: key -> (dim,ndim,dtype)
        :return: whether we added a new entry
        :rtype: bool
        """
        from returnn.util.basic import hdf5_strings

        added_count = 0
        for data_key, (dim, ndim, dtype) in extra_type.items():
            assert data_key != "inputs"
            if data_key in self._prepared_extra:
                return
            if not self._prepared_extra and not self.extend_existing_file:
                # For the first time, need to create the groups.
                self._file.create_group("targets/data")
                self._file.create_group("targets/size")
                self._file.create_group("targets/labels")
            hdf5_strings(self._file, "targets/labels/%s" % data_key, ["dummy-label"])
            if ndim == 0:
                ndim = 1  # we will automatically add a dummy-dim
            shape = [None] * ndim  # type: typing.List[typing.Optional[int]]
            if ndim >= 2:
                shape[-1] = dim
            if dtype == "string":
                # noinspection PyUnresolvedReferences
                dtype = h5py.special_dtype(vlen=str)
            if self.extend_existing_file:
                self._datasets[data_key] = self._file["targets/data"][data_key]
                assert shape[0] is None
                self._extra_num_time_steps[data_key] = self._datasets[data_key].shape[0]
            else:
                self._datasets[data_key] = self._file["targets/data"].create_dataset(
                    data_key, shape=[d if d else 0 for d in shape], dtype=dtype, maxshape=shape
                )
                self._file["targets/size"].attrs[data_key] = [dim or 1, ndim]
                self._extra_num_time_steps[data_key] = 0
            self._prepared_extra.add(data_key)
            added_count += 1
        if added_count and not self.extend_existing_file:
            assert self._prepared_extra
            self._seq_lengths.resize(1 + len(self._prepared_extra), axis=1)
        return bool(added_count)

    def _insert_h5_inputs(self, raw_data):
        """
        Inserts a record into the hdf5-file.
        Resizes if necessary.

        :param numpy.ndarray raw_data: shape=(time,data) or shape=(time,)
        """
        assert raw_data.ndim >= 1
        name = "inputs"
        if self.extend_existing_file:
            # Just expect that the same dataset already exists.
            self._datasets[name] = self._file[name]
        if name not in self._datasets:
            self._datasets[name] = self._file.create_dataset(
                name, raw_data.shape, raw_data.dtype, maxshape=tuple(None for _ in raw_data.shape)
            )
        else:
            old_shape = self._datasets[name].shape
            self._datasets[name].resize(old_shape[0] + raw_data.shape[0], axis=0)
        # append raw data to dataset
        self._datasets[name][self._file.attrs["numTimesteps"] :] = raw_data
        self._file.attrs["numTimesteps"] += raw_data.shape[0]
        self._file.attrs["numSeqs"] += 1

    def _insert_h5_other(self, data_key, raw_data, dtype=None, add_time_dim=False, dim=None):
        """
        :param str data_key:
        :param numpy.ndarray|int|float|list[int]|numpy.float32|numpy.int32 raw_data:
          shape=(time,data) or shape=(time,) or shape=()...
        :param str|None dtype:
        :param bool add_time_dim:
        :param int|None dim:
        """
        if isinstance(raw_data, (int, float, list, numpy.float32, numpy.int32)):
            raw_data = numpy.array(raw_data)
        assert isinstance(raw_data, numpy.ndarray), "raw_data is %r of type %r" % (raw_data, type(raw_data))
        if add_time_dim or raw_data.ndim == 0:
            raw_data = numpy.expand_dims(raw_data, 0)
        assert raw_data.ndim > 0 and raw_data.shape[0] > 0
        if dtype:
            raw_data = raw_data.astype(dtype)
        if dim is None:
            if raw_data.ndim > 1:
                dim = raw_data.shape[-1]
            else:
                dim = 1  # dummy

        # We assume that _insert_h5_inputs was called before.
        assert self._file.attrs["numSeqs"] > 0 and self._seq_lengths.shape[0] > 0
        seq_idx = self._file.attrs["numSeqs"] - 1

        if raw_data.dtype == object:
            # Is this a string?
            assert isinstance(raw_data.flat[0], (str, bytes))
            dtype = "string"
        else:
            dtype = raw_data.dtype.name
        if self._prepare_extra({data_key: (dim, raw_data.ndim, dtype)}):
            # We added it now. Maybe other extra data keys were added before. The data_key_idx is different now.
            # Thus, seq_lengths might have become invalid. Reinit them.
            assert seq_idx == 0 or self.extend_existing_file  # We can only do that in the beginning.
            for data_key_idx_0, data_key_ in enumerate(sorted(self._prepared_extra)):
                self._seq_lengths[seq_idx, data_key_idx_0 + 1] = self._extra_num_time_steps[data_key_]

        self._extra_num_time_steps[data_key] += raw_data.shape[0]
        self._datasets[data_key].resize(self._extra_num_time_steps[data_key], axis=0)

        data_key_idx = sorted(self._prepared_extra).index(data_key) + 1
        self._seq_lengths[seq_idx, data_key_idx] = raw_data.shape[0]

        offset = self._extra_num_time_steps[data_key] - raw_data.shape[0]
        hdf_data = self._datasets[data_key]
        hdf_data[offset:] = raw_data

    def insert_batch(self, inputs, seq_len, seq_tag, extra=None):
        """
        :param numpy.ndarray inputs: shape=(n_batch,time,data) (or (n_batch,time), or (n_batch,time1,time2), ...)
        :param list[int]|dict[int,list[int]|numpy.ndarray] seq_len: sequence lengths (per axis, excluding batch axis)
        :param list[str|bytes] seq_tag: sequence tags of length n_batch
        :param dict[str,numpy.ndarray]|None extra: one or multiple possible targets data.
            key can be "classes" or anything.
            The dtype and dim is inferred automatically from the Numpy array.
            If there are multiple items, the seq length must be the same currently.
            Must be batch-major, and following the time, then the feature.
        """
        n_batch = len(seq_tag)
        assert n_batch == inputs.shape[0]
        assert inputs.ndim == self.ndim + 1  # one more for the batch-dim
        if not isinstance(seq_len, dict):
            seq_len = {0: seq_len}
        assert isinstance(seq_len, dict)
        assert all(
            [isinstance(key, int) and isinstance(value, (list, numpy.ndarray)) for (key, value) in seq_len.items()]
        )
        if seq_len:
            ndim_with_seq_len = max(seq_len.keys()) + 1
        else:
            ndim_with_seq_len = 0
        sparse = ndim_with_seq_len == self.ndim
        assert ndim_with_seq_len <= self.ndim
        assert all([0 <= key < ndim_with_seq_len for key in seq_len.keys()])
        assert len(seq_len) == ndim_with_seq_len
        assert all([n_batch == len(value) for (key, value) in seq_len.items()])
        assert all([max(value) == inputs.shape[key + 1] for (key, value) in seq_len.items()])
        if self.dim and not sparse:
            assert self.dim == inputs.shape[-1]
        if extra:
            assert all([n_batch == value.shape[0] for value in extra.values()]), "n_batch %i, extra shapes: %r" % (
                n_batch,
                {key: value.shape for (key, value) in extra.items()},
            )

        seqlen_offset = self._seq_lengths.shape[0]
        self._seq_lengths.resize(seqlen_offset + n_batch, axis=0)
        self._seq_tags.resize(seqlen_offset + n_batch, axis=0)

        for i in range(n_batch):
            self._seq_tags[seqlen_offset + i] = numpy.array(seq_tag[i], dtype=self._seq_tags.dtype)
            # Note: Currently, our HDFDataset does not support to have multiple axes with dynamic length.
            # Thus, we flatten all together, and calculate the flattened seq len.
            # (Ignore this if there is only a single time dimension.)
            flat_seq_len = int(numpy.prod([seq_len[axis][i] for axis in range(ndim_with_seq_len)]))
            assert flat_seq_len > 0
            flat_shape = [flat_seq_len]
            if self.dim and not sparse:
                flat_shape.append(self.dim)
            self._seq_lengths[seqlen_offset + i, 0] = flat_seq_len
            data = inputs[i]
            data = data[tuple([slice(None, seq_len[axis][i]) for axis in range(ndim_with_seq_len)])]
            data = numpy.reshape(data, flat_shape)
            self._insert_h5_inputs(data)
            if len(seq_len) > 1:
                # Note: Because we have flattened multiple axes with dynamic len into a single one,
                # we want to store the individual axes lengths. We store those in a separate data entry "sizes".
                # Note: We could add a dummy time-dim for this "sizes", and then have a feature-dim = number of axes.
                # However, we keep it consistent to how we handled it in our 2D MDLSTM experiments.
                self._insert_h5_other(
                    "sizes", [seq_len[axis][i] for axis in range(ndim_with_seq_len)], add_time_dim=False, dtype="int32"
                )
            if extra:
                try:
                    for key, value in extra.items():
                        assert value.shape[0] == n_batch
                        self._insert_h5_other(key, value[i])
                except Exception:
                    print(
                        "%s: insert extra exception. input shape %r, seq len %r, extra shapes: %r"
                        % (
                            self,
                            inputs.shape,
                            seq_len,
                            {
                                key: value.shape if isinstance(value, numpy.ndarray) else repr(value)
                                for (key, value) in extra.items()
                            },
                        ),
                        file=log.v3,
                    )
                    raise

    def close(self):
        """
        Closes the file.
        """
        import os
        import shutil

        if self._file:
            self._file.close()
            self._file = None
        if self.tmp_filename:
            if not self.extend_existing_file:
                assert not os.path.exists(self.filename)
                # Otherwise we have made sure that the existing file was copied and expanded.
            tmp_dest_filename = "%s/.%s.copying" % (
                os.path.dirname(self.filename) or ".",
                os.path.basename(self.filename),
            )
            shutil.copyfile(self.tmp_filename, tmp_dest_filename)
            shutil.move(tmp_dest_filename, self.filename)
            os.remove(self.tmp_filename)
            self.tmp_filename = None


class HDFDatasetWriter:
    """
    Similar as :class:`SimpleHDFWriter`, but is mostly intended to copy an existing dataset,
    see :func:`dump_from_dataset`.
    The resulting HDF file can be read later by :class:`HDFDataset`.
    """

    def __init__(self, filename):
        """
        :param str filename: for the HDF to write
        """
        print("Creating HDF dataset file %s" % filename, file=log.v3)
        self.filename = filename
        self.file = h5py.File(filename, "w")

    def close(self):
        """
        Close the HDF file.
        """
        self.file.close()

    def dump_from_dataset(self, dataset, epoch=1, start_seq=0, end_seq=float("inf"), use_progress_bar=True):
        """
        :param Dataset dataset: could be any dataset implemented as child of Dataset
        :param int epoch: for dataset
        :param int start_seq:
        :param int|float end_seq:
        :param bool use_progress_bar:
        """
        from returnn.util.basic import NumbersDict, human_size, progress_bar_with_time, try_run

        hdf_dataset = self.file

        print("Work on epoch: %i" % epoch, file=log.v3)
        dataset.init_seq_order(epoch)

        data_keys = sorted(dataset.get_data_keys())
        assert data_keys, "Got no data keys from dataset to write to HDF."
        print("Data keys:", data_keys, file=log.v3)
        if "orth" in data_keys:  # special workaround for now, not handled
            data_keys.remove("orth")
        if "raw" in data_keys:
            data_keys.remove("raw")
        data_target_keys = [key for key in dataset.get_target_list() if key in data_keys]
        data_input_keys = [key for key in data_keys if key not in data_target_keys]
        default_data_input_key = None
        if data_input_keys:
            if len(data_input_keys) > 1:
                if "data" in data_input_keys:
                    default_data_input_key = "data"
                else:
                    raise Exception("not sure which input data key to use from %r" % (data_input_keys,))
            else:
                default_data_input_key = data_input_keys[0]
            progress_bar_data_key = default_data_input_key
        else:
            progress_bar_data_key = "classes" if "classes" in data_target_keys else data_target_keys[0]
        print("Using input data key:", default_data_input_key)

        # All but one of the inputs have to be treated as targets because our HDF format only supports one input.
        data_target_keys += [key for key in data_input_keys if key != default_data_input_key]
        data_input_key = default_data_input_key

        hdf_data_key_map = {key: key for key in data_keys if key != data_input_key}
        if "data" in hdf_data_key_map:
            hdf_data_key_map["data"] = "classes"  # Replace "data" which is reserved for input key in HDFDataset.
            assert "classes" not in hdf_data_key_map

        # We need to do one run through the dataset to collect some stats like total len.
        print("Collect stats, iterate through all data...", file=log.v3)
        seq_idx = start_seq
        seq_idxs = []
        seq_tags = []
        seq_lens = []
        total_seq_len = NumbersDict(0)
        max_tag_len = 0
        dataset_num_seqs = try_run(lambda: dataset.num_seqs, default=None)  # can be unknown
        if end_seq != float("inf"):
            if dataset_num_seqs is not None:
                dataset_num_seqs = min(dataset_num_seqs, end_seq)
            else:
                dataset_num_seqs = end_seq
        if dataset_num_seqs is not None:
            dataset_num_seqs -= start_seq
            assert dataset_num_seqs > 0
        while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= end_seq:
            seq_idxs += [seq_idx]
            dataset.load_seqs(seq_idx, seq_idx + 1)
            seq_len = dataset.get_seq_length(seq_idx)
            seq_lens += [seq_len]
            tag = dataset.get_tag(seq_idx)
            seq_tags += [tag]
            max_tag_len = max(len(tag), max_tag_len)
            total_seq_len += seq_len
            if use_progress_bar and dataset_num_seqs is not None:
                progress_bar_with_time(float(seq_idx - start_seq) / dataset_num_seqs)
            seq_idx += 1
        num_seqs = len(seq_idxs)

        assert num_seqs > 0
        shapes = {}
        for data_key in data_keys:
            assert data_key in total_seq_len.dict
            shape = [total_seq_len[data_key]]
            shape += dataset.get_data_shape(data_key)
            print(
                "Total len of %r is %s, shape %r, dtype %s"
                % (data_key, human_size(shape[0]), shape, dataset.get_data_dtype(data_key)),
                file=log.v3,
            )
            shapes[data_key] = shape

        print("Set seq tags...", file=log.v3)
        hdf_dataset.create_dataset("seqTags", shape=(num_seqs,), dtype="S%i" % (max_tag_len + 1))
        for i, tag in enumerate(seq_tags):
            hdf_dataset["seqTags"][i] = numpy.array(tag, dtype="S%i" % (max_tag_len + 1))
            if use_progress_bar:
                progress_bar_with_time(float(i) / num_seqs)

        print("Set seq len info...", file=log.v3)
        hdf_dataset.create_dataset(HDF_SEQ_LENGTHS_KEY, shape=(num_seqs, len(data_keys)), dtype="int32")
        for i, seq_len in enumerate(seq_lens):
            data_len = [seq_len[data_input_key]] if data_input_key else []
            targets_lens = [seq_len[data_key] for data_key in sorted(data_target_keys)]
            hdf_dataset[HDF_SEQ_LENGTHS_KEY][i] = data_len + targets_lens
            if use_progress_bar:
                progress_bar_with_time(float(i) / num_seqs)

        print("Create arrays in HDF...", file=log.v3)
        hdf_dataset.create_group("targets/data")
        hdf_dataset.create_group("targets/size")
        hdf_dataset.create_group("targets/labels")
        for data_key in data_keys:
            if data_input_key and data_key == data_input_key:
                hdf_dataset.create_dataset("inputs", shape=shapes[data_key], dtype=dataset.get_data_dtype(data_key))
            else:
                hdf_dataset["targets/data"].create_dataset(
                    hdf_data_key_map[data_key], shape=shapes[data_key], dtype=dataset.get_data_dtype(data_key)
                )
                hdf_dataset["targets/size"].attrs[hdf_data_key_map[data_key]] = dataset.num_outputs[data_key]
            if data_key in dataset.labels:
                labels = dataset.labels[data_key]
                labels = [label.encode("utf8") for label in labels]
                assert len(labels) == dataset.num_outputs[data_key][0]
            else:
                labels = ["%s-class-%i" % (data_key, i) for i in range(dataset.get_data_dim(data_key))]
            print("Labels for %s:" % data_key, labels[:3], "...", file=log.v5)
            max_label_len = max(map(len, labels))
            if not data_input_key or data_key != data_input_key:
                hdf_dataset["targets/labels"].create_dataset(
                    hdf_data_key_map[data_key], (len(labels),), dtype="S%i" % (max_label_len + 1)
                )
                for i, label in enumerate(labels):
                    hdf_dataset["targets/labels"][hdf_data_key_map[data_key]][i] = numpy.array(
                        label, dtype="S%i" % (max_label_len + 1)
                    )

        # Again iterate through dataset, and set the data
        print("Write data...", file=log.v3)
        dataset.init_seq_order(epoch)
        offsets = NumbersDict(0)
        for seq_idx, tag in zip(seq_idxs, seq_tags):
            dataset.load_seqs(seq_idx, seq_idx + 1)
            tag_ = dataset.get_tag(seq_idx)
            assert tag == tag_  # Just a check for sanity. We expect the same order.
            seq_len = dataset.get_seq_length(seq_idx)
            for data_key in data_keys:
                if data_input_key and data_key == data_input_key:
                    hdf_data = hdf_dataset["inputs"]
                else:
                    hdf_data = hdf_dataset["targets/data"][hdf_data_key_map[data_key]]
                data = dataset.get_data(seq_idx, data_key)
                hdf_data[offsets[data_key] : offsets[data_key] + seq_len[data_key]] = data

            if use_progress_bar:
                progress_bar_with_time(float(offsets[progress_bar_data_key]) / total_seq_len[progress_bar_data_key])

            offsets += seq_len

        assert offsets == total_seq_len  # Sanity check.

        # Set some old-format attribs. Not needed for newer RETURNN versions.
        hdf_dataset.attrs[HDF_INPUT_PATT_SIZE_KEY] = dataset.num_inputs

        print("All done.", file=log.v3)
