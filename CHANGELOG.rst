Changelog
=========

Current Master (0.3+git)
------------------------

- Added temporary file cache manager (activated via `use_cache_manager` as for the i6 specific caching, `<https://github.com/JackTemaki/MiniReturnn/pull/13>`_)
- Small fix for dataset initialization which avoids unnecessary re-init (`<https://github.com/JackTemaki/MiniReturnn/pull/14>`_)
- Refactor of Chunking mechanism and definition (`<https://github.com/JackTemaki/MiniReturnn/pull/15>`_)
- Refactor updater.py (docstrings, deprecate using callables, typing)
- Add thread locking for cache manager (`https://github.com/JackTemaki/MiniReturnn/pull/16_`)
- Merge torch average checkpoint script from upstream (b346ef0 -> 120d28c, including fix e6f3f5d->d12a59f)
- HDFDataset sparse fix from upstream (`<https://github.com/rwth-i6/returnn/pull/1503>`_)
- uint16 type fix from upstream (`<https://github.com/rwth-i6/returnn/pull/1488>`_)
- Remove more unused code, e.g. in logging, debug, removed unused Gammatone code
- Deleted old docs, added simple configuration.rst file

Version 0.3
-----------

This is the last version that is kept compatible with mainline RETURNN, roughly at the state of late October 2023.

All following version will deviate.

- Further various code cleanup (`<https://github.com/JackTemaki/MiniReturnn/pull/9>`_):
    - remove commented code in __main__ and make not-implemented messages more verbose
    - remove dead code in datastes/util/vocabulary.py
    - remove unused torch/tensor_utils.py
    - remove unused horovod code
    - refactor HDF datasets: split NextGenHDF into a new file and rename to StreamHDFDataset, move HDF writer helpers to util/
    - remove normalization dataset and siamese dataset
- merge OggZipDataset Test from upstream
- allow for ndarray containing strings in ``create_tensor``
- Do not move Tensor to target device in ``collate_batch``, which caused that the dataloader threads reserves GPU memory
- Enable multiprocessing in dataloading (`<https://github.com/JackTemaki/MiniReturnn/pull/11>`_)
    - Introduce `num_workers_for_gpu` flag which allows for multiprocessing with the PT Dataloader
    - Introduce (automatic) sharding for RETURNN Datasets (For now Generating and MetaDataset) to be able to use multiple dataloader workers
    - Remove unneeded MultiProcDataset
- Small changes from upstream
  - Fix in TranslationDataset for Pickling (6eb04d2 -> 2379615)
  - no init_seq_order in OggZip constructor (11d3346 -> e3f375b)
  - Allow unexpected keys (809a649 -> 9db8fd2)
- Enable gradient accumulation (`<https://github.com/JackTemaki/MiniReturnn/pull/12>`_)


Version 0.2
-----------

- provide ``epoch`` and ``step`` in ``RunCtx`` (`<https://github.com/JackTemaki/MiniReturnn/pull/4>`_)
- ``allow_missing_optimizer_checkpoint`` config parameter to allow the usage of a fresh optimizer in case the optimizer checkpoint for the chosen epoch can't be found
- use ``persistent_worker=True`` in ``DataLoader`` to prohibit premature deletion of Cuda Tensors within the loader process
- merge upstream https://github.com/rwth-i6/returnn/pull/1347 (fix for MetaDataset)
- merge upstream https://github.com/rwth-i6/returnn/pull/1344 (min_seq_len/max_seq_len support)
- merge upstream https://github.com/rwth-i6/returnn/pull/1346 (load model to correct device)
- merge upstream https://github.com/rwth-i6/returnn/pull/1358 (min_chunk_size parameter)
- add ``tools/torch_export_to_onnx.py``
- merge upstream https://github.com/rwth-i6/returnn/pull/1364 (fix in cleanup_models regarding learning rate control)
- fix missing run_ctx init in onnx export
- integrate gradient clipping/norm: https://github.com/JackTemaki/MiniReturnn/pull/6
- print CUDA memory information
- small fix for checkpoint loading: https://github.com/JackTemaki/MiniReturnn/pull/8


Version 0.1
-----------

This part is taken from the readme at time of tagging v0.1.

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


Added features that are likely to appear in mainline-RETURNN:
 - Checkpoint cleanup, currently pending for mainline RETURNN in https://github.com/rwth-i6/returnn/pull/1316
 - seq_tag, seq_idx and non-Tensor data support in the data pipeline, pending at: https://github.com/rwth-i6/returnn/pull/1330


Experimental features that might not be needed:
 - ``batching_drop_last`` config parameter to discard the last incomplete batch in an epoch
 - forward init/finish hooks that can be used to attach custom objects to the run_ctx
