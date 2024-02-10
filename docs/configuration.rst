=============
Configuration
=============

The configuration file for Returnn should contain executable python code.
All parameters are accessed from the defined top-level variables.


Mandatory Parameters
--------------------

task
    The task, such as ``"train"`` or ``"forward"``. This will determine in which mode RETURNN runs.

device
    A string being ``"cuda"`` (alternatively ``"gpu"``) or ``"cpu"``.

train / dev
    The datasets parameters are set to a python dict with a mandatory entry ``"class"``.
    The ``"class"`` attribute needs to be set to the class name of the dataset that should be used.
    It is recommended to always use a ``"MetaDataset"`` as top-level dataset.
    ``train`` and ``dev`` are used during training for the training and cross-validation data respectively.
    See `here <https://github.com/JackTemaki/MiniReturnn/tree/master/returnn/datasets>`_ for the datasets,
    and in particular have a look at the ``"MetaDataset"`` in ``meta.py``.

    Optionally you can define "eval_dataset" of type ``Dict[str, <datset_dict>]`` for additional datasets
    to be run after each training epoch.


batch_size
    The total number of frames, samples or tokens. A mini-batch has at least a time-dimension
    and a batch-dimension (or sequence-dimension), and depending on dense or sparse,
    also a feature-dimension.
    ``batch_size`` is the upper limit for ``time * sequences`` during creation of the mini-batches.

max_seqs
    The maximum number of sequences in one mini-batch., together with ``batch_size=None`` you can define a static batch size.

learning_rates
    A list of learning rates for each epoch, if not fully specified the last value is used for the remaining epochs.

optimizer
    A dictionary defining the optimizer options. The ``class`` is the only mandatory field,
    which should contain the case-insensitive name of the PyTorch optimizer module name (e.g. "adam").

num_epochs
    The number of epochs to train.

log_verbosity
    An integer. Common values are 3 or 4. Starting with 5, you will get an output per mini-batch.

get_model
    A callable that returns a torch.nn.Module instance to be used as the main NN model.
    Can be the class definition of a module itself, as this is technically also a callable (constructor).

learning_rate_file
    A path to a file storing the learning rate for each epoch. Despite the name, also stores scores and errors values.


Training
--------

chunking
    Chunks each sequence into smaller parts and puts each part individually into the batch.

cleanup_old_models
    If set to ``True``, checkpoints are removed based on their score on the dev set.
    Per default, 2 recent, 4 best, and the checkpoints 20,40,80,160,240 are kept.
    Can be set as a dictionary to specify additional options:

    - ``keep_last_n``: integer defining how many recent checkpoints to keep
    - ``keep_best_n``: integer defining how many best checkpoints to keep
    - ``keep``: list or set of integers defining which checkpoints to keep

max_seq_length
    A dict with string:integer pairs. The string must be a valid data key,
    and the integer specifies the upper bound for this data object.
    During batch construction any sequence where the specified data object exceeds the upper bound are discarded.
    Note that some datasets (e.g ``OggZipDataset``) load and process the data
    to determine the length, so even for discarded sequences data processing might be performed.

save_interval
    An integer specifying after how many epochs the model is saved.

stop_on_nonfinite_train_score
    If set to ``False``, the training will not be interrupted if a single update step has a loss with NaN of Inf,
    but the update step will simply be ignored.


Updater Settings
----------------

accum_grad_multiple_step
    An integer specifying the number of updates to stack the gradient, called "gradient accumulation".

gradient_clip
    Specify a gradient clipping threshold.

gradient_noise
    Apply a (gaussian?) noise to the gradient with given deviation (variance? stddev?)

learning_rate_control
    This defines which type of learning rate control mechanism is used. Possible values are:

    - ``constant`` for a constant learning rate which is never modified
    - ``newbob_abs`` for a scheduling based on absolute improvement
    - ``newbob_rel`` for a scheduling based on relative improvement
    - ``newbob_multi_epoch`` for a scheduling based on relative improvement averaged over multiple epochs

    Please also look at setting values with the ``newbob`` prefix for further customization


Model Loading
-------------

.. note::

    This documentation does not cover all possible combinations of parameters for loading models.
    For more details, please refer to
    `EngineBase <https://github.com/rwth-i6/returnn/blob/master/returnn/engine/base.py>`_
    and also check
    :class:`CustomCheckpointLoader <returnn.tf.network.CustomCheckpointLoader>`
    and the
    `"Different ways to import parameters" <https://github.com/rwth-i6/returnn/wiki/Different-ways-to-import-parameters>`_
    page.

allow_random_model_init
    Initialize a model randomly.
    This can be useful if you want to use only ``preload_from_files`` to load
    multiple different models into one config for decoding without using ``load``.

import_model_train_epoch1
    If a path to a valid model is provided
    (for TF models paths with or without ``.meta`` or ``.index`` extension are possible),
    use this to initialize the weights for training.
    If you do not want to start a new training, see ``load``.

load
    If a path to a valid model is provided
    (for TF models paths with or without ``.meta`` or ``.index`` extension are possible),
    use this to load the specified model and training state.
    The training is continued from the last position.

load_epoch
    Specifies the epoch index, and selects the checkpoint based on the prefix given in ``model``.
    If not set, RETURNN will determine the epoch from the filename or use the latest epoch in case
    of providing only ``model``.

preload_from_files
    A dictionary that contains a ``filename`` entry and optional parameters to define specific model loading.
    If ``prefix`` is defined, it will load the parameters from the checkpoint but only replace the layers that start
    with the given prefix. The layer name in the checkpoint should match the name of the layer without the prefix
    (e.g. the parameters of "submodel1_layer1" in the network would be set to the parameters of "layer1" in the
    checkpoint).
    Example (containing all possible parameters)::

        preload_from_files = {
          "existing-model": {
            "filename": ".../net-model/network.163",  # your checkpoint file, mandatory
            "init_for_train": True,  # only load the checkpoint at the start of training epoch 1, default is False
            "ignore_missing": True,  # if the checkpoint only partly covers your model, default is False
            "ignore_params": ["some_parameter", ...],  # list of parameter names that should not be loaded
            "ignore_params_prefixes": ["some_prefix_", ...],  # list of parameter prefixes that should not be loaded
            "var_name_mapping": {"name_in_graph": "name_in_checkpoint", ...},  # map non-matching parameter names
            "prefix": "submodel1_",  # only load parameters for layers starting with the given prefix
          }
        }

load_ignore_missing_vars
    If enabled, it will ignore missing variables when loading a checkpoint.
    Otherwise it will error on missing variables.
    Non-loaded variables are using the standard variable initialization (e.g. random init).
    By default, this is disabled.


Dynamic Learning Rate Settings
------------------------------

learning_rate_control_error_measure
    A str to define which score or error is used to control the learning rate reduction.
    Per default, Returnn will use dev_score_output.
    A typical choice would be dev_score_LAYERNAME or dev_error_LAYERNAME.
    Can be set to None to disable learning rate control.

learning_rate_control_min_num_epochs_per_new_lr
    The number of epochs after the last update that the learning rate is kept constant.

learning_rate_control_relative_error_relative_lr
    If true, the relative error is scaled with the ratio of the default learning rate divided by the current
    learning rate.
    Can be used with ``newbob_rel`` and ``newbob_multi_epoch``.

min_learning_rate
    Specifies the minimum learning rate.

newbob_error_threshold
    This is the absolute improvement that has to be achieved in order to _not_ reduce the learning rate.
    Can be used with ``newbob_abs``.
    The value can be positive or negative.

newbob_learning_rate_decay
    The scaling factor for the learning rate when a reduction is applied.
    This parameter is available for all ``newbob`` variants.

newbob_multi_num_epochs
    The number of epochs the improvement is averaged over.

newbob_multi_update_interval
    The number of steps after which the learning rate is updated.
    This is set equal to ``newbob_multi_num_epochs`` when not specified.

newbob_relative_error_threshold
    This is the relative improvement that has to be achieved in order to _not_ reduce the learning rate.
    Can be used with ``newbob_rel`` and ``newbob_multi_epoch``.
    The value can be positive or negative.

relative_error_div_by_old
    If true the relative error is computed by dividing the error difference by the old error value instead of the
    current error value.
