Changelog
=========



Current Master (0.1+git)
------------------------

- provide ``epoch`` and ``step`` in ``RunCtx`` (`<https://github.com/JackTemaki/MiniReturnn/pull/4>`_)
- ``allow_missing_optimizer_checkpoint`` config parameter to allow the usage of a fresh optimizer in case the optimizer checkpoint for the chosen epoch can't be found

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
