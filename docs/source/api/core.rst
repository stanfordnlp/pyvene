``pyvene``: Core API
=========================================

.. module:: pyvene
.. currentmodule:: pyvene

.. toctree::
    :hidden:

    interventions
    wrappers
    models


Core model wrappers
-------------------

Whenever we want to wrap a model with intervention capabilities, the following classes are used:

.. autosummary::

  IntervenableConfig
  RepresentationConfig
  IntervenableModel
  IntervenableNdifModel
  build_intervenable_model

Interventions
-------------

Pyvene supports many types of interventions out-of-the-box. They are:

.. autosummary::
  CollectIntervention
  ZeroIntervention
  VanillaIntervention
  AdditionIntervention
  SubtractionIntervention
  NoiseIntervention
  RotatedSpaceIntervention
  BoundlessRotatedSpaceIntervention
  SigmoidMaskRotatedSpaceIntervention
  LowRankRotatedSpaceIntervention
  PCARotatedSpaceIntervention
  SigmoidMaskIntervention
  AutoencoderIntervention

Additionally, we have abstract classes that you may use to create your own interventions:

.. autosummary::
  Intervention
  LocalistRepresentationIntervention
  DistributedRepresentationIntervention
  TrainableIntervention
  ConstantSourceIntervention
  SourcelessIntervention
  BasisAgnosticIntervention
  SharedWeightsTrainableIntervention
  SkipIntervention


Built-in model types
--------------------

Pyvene supports many types of models out-of-the-box. There are two dictionaries which
record useful information about these models:

.. autosummary::
  type_to_module_mapping
  type_to_dimension_mapping

We provide the following model types with unified helper names for their internal components:

.. autosummary::
  create_gpt2
  create_gpt2_lm
  create_blip
  create_blip_itm
  create_gpt_neo
  create_gpt_neox
  create_gru
  create_gru_lm
  create_gru_classifier
  create_llava
  create_llama
  create_mlp_classifier
  create_backpack_gpt2