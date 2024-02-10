=======================
Welcome to Mini-RETURNN
=======================

Mini-RETURNN is a feature-less derivative of `RETURNN <https://github.com/rwth-i6/returnn>`__ which only contains of basic components needed to train sequence-to-sequence models (e.g. ASR, MT or TTS) with PyTorch.
No network helper functions are provided, all model logic has to be explicitly defined by the user.
This is repository is intended to be a quick playground for custom experiments, for everything serious please use `RETURNN <https://github.com/rwth-i6/returnn>`__.

Mini-RETURNN is not intended to be fully compatible to Mainline RETURNN. It is supposed to be a lightweight
alternative purely focused on PyTorch and academic research. For an overview on changes see the last section below.

General config-compatibility to RETURNN is kept, especially with respect to Sisyphus integration.


Installation
------------

PyTorch >= 1.13 is recommended, but might not be strictly necessary.
You can use the `requirements.txt` file to install all strictly necessary packages.
Additional packages might be required for certain features, which can be installed using `requirements-optional.txt`.

Usage
-----

Mini-RETURNN is intended to be used in conjunction with the i6-style Sisyphus experiment pipelines.
A current example setup can be found `here <https://github.com/rwth-i6/i6_experiments/tree/main/users/rossenbach/experiments/rescale/tedlium2_standalone_2023>`_, but a proper example setup will be added in the future.

In general, the ``rnn.py`` file is intended to be the entry point in using RETURNN, used in combination with a config file:
``python3 rnn.py <config_file>``


Difference to Mainline RETURNN
------------------------------

Important differences to mainline RETURNN:

Removed features:
 - Anything related to the Tensorflow backend (also tools and tests)
 - Anything related to the Frontend API
 - Window/Chunking/Batching logic WITHIN DATASETS (Batching and Chunking exists in the new PyTorch datapipeline)
 - Some older Datasets that depended on removed features (no relevant Dataset should be missing)
 - Most utility code that was only used by Tensorflow code
 - There is no default keep-pattern of checkpoints, ``keep`` has to specified within the ``cleanup_old_models`` config dict explicitely
 - "eval" dataset is no longer allowed, use "eval_datasets" instead
 - ``__main__.py`` no longer handles datasets
 - "hdf_dump" no longer allows strings but only config files, dumps only "train" (which it probably also did before).


Not yet added features:
 - Multi-GPU training
 - Many small changes around the PyTorch Engine and other parts


Changed behavior:
 - The data to the actual step function in the config is passed as PyTorch tensor dict instead of Frontend Tensors
   - Axis information is automatically added as ``<data_name>:size<idx>`` entry starting from 0 = batch axis
   - Axis information is always placed on the target device
 - The Loss class has less parameters, e.g. ``use_normalization`` does not exist, and the behavior is always true.
   -  Also determining the inverse norm factor automatically is not possible, it has to be provided explictly
 - The Engine API is regarding step functions is structured slightly differently
 - Step-logging is slightly differently
 - Overriding the Engine by declaring a ``CustomEngine`` in the config is possible, see https://github.com/rwth-i6/returnn/pull/1306 for a discussion on this.
 - ``weight_decay`` is applied to ALL parameters without exception, some discussion in https://github.com/rwth-i6/returnn/issues/1319 ,
   although the conclusion that the mainline RETURNN behavior can be non-deterministic was not reached there.
 - Always uses the cache-manager if available, even when not running in cluster
 - seq_tag, seq_idx and non-Tensor/np.array data support works differently
 - forward init/finish hook interface is different

Experimental features that might not be needed:
 - ``batching_drop_last`` config parameter to discard the last incomplete batch in an epoch

