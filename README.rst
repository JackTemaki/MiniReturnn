=======================
Welcome to Mini-RETURNN
=======================

Mini-RETURNN is a feature-less derivative of `RETURNN <https://github.com/rwth-i6/returnn>`__ which only contains of basic components needed to train sequence-to-sequence models (e.g. ASR, MT or TTS) with PyTorch.
No network helper functions are provided, all model logic has to be explicitly defined by the user.
This is repository is intended to be a quick playground for custom experiments, for everything serious please use `RETURNN <https://github.com/rwth-i6/returnn>`__.

General config-compatibility to RETURNN is kept, but technical details differ, especially regarding the Torch engine class.
Nevertheless, expect the Mini-RETURNN config to be more strict and more verbose, with less implicit or default assumptions.


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
 - Dataloader2 from ``torchdata`` was replaced by Dataloader from ``torch.utils.data``, as Dataloader2 has a non-stable API. In addition, num_workers=1 with "spawn" multiprocessing is set. This means that an extra process loads the data, and prefetch is working correctly, resulting in significant speedups.
 - seq_tag, seq_idx and non-Tensor/np.array data support works differently
 - forward init/finish hook interface is different (might be streamlined in a future version)

Additional features:
 - gradient clipping by norm or value

Experimental features that might not be needed:
 - ``batching_drop_last`` config parameter to discard the last incomplete batch in an epoch



Installation
------------

PyTorch >= 1.13 is recommended, but might not be strictly necessary.
You can use the `requirements.txt` file to install all strictly necessary packages.
Additional packages might be required for certain features, which can be installed using ``requirements-optional.txt`.

