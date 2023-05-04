==================
Welcome to Mini-RETURNN
==================

Mini-RETURNN is a feature-less derivative of `RETURNN <https://github.com/rwth-i6/returnn>`__ which only contains of basic components needed to train sequence-to-sequence models (e.g. ASR, MT or TTS) with PyTorch.
No network helper functions are provided, all model logic has to be explicitly defined by the user.
This is repository is intended to be a quick playground for custom experiments, for everything serious please use `RETURNN <https://github.com/rwth-i6/returnn>`__.

General config-compatibility to RETURNN is kept, but technical details differ.


Removed features:
 - Anything related to the Tensorflow backend (also tools and tests)
 - Anything related to the Frontend API
 - Window/Chunking/Batching logic WITHIN DATSETS (Batching and Chunking exists in the new PyTorch datapipeline)
 - Some older Datasets that depended on removed features (no relevant Dataset should be missing)
 - Most utility code that was only used by Tensorflow code
 - There is no default keep-pattern of checkpoints, `keep` has to specified within the `cleanup_old_models` config dict explicitely


Changed behavior
 - The data to the actual step function in the config is passed as PyTorch tensor dict instead of Frontend Tensors
   - Axis information is automatically added as <data_name>:axis<idx> entry starting from 0 = batch axis
 - The Loss class has less parameters, e.g. `use_normalization` does not exist, and the behavior is always true.
   -  Also determining the inverse norm factor automatically is not possible, it has to be provided explictly
 - The Engine API is regarding step functions is structured slightly differently
 - Step-logging is slightly differently
 - Overriding the Engine by declaring a `CustomEngine` in the config is possible, see https://github.com/rwth-i6/returnn/pull/1306 for a discussion on this.
 - `weight_decay` is applied to ALL parameters without exception.
 - Always uses the cache-manager if available, even when not running in cluster
