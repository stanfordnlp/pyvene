.. pyvene documentation master file, created by
   sphinx-quickstart on Fri Jul 12 16:49:16 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. container::

   .. rubric:: |image1|
      :name: section

   `Read Our Paper » <https://arxiv.org/abs/2403.07809>`__


pyvene
======

|image2| |image3| |image4|

.. |image1| image:: https://i.ibb.co/BNkhQH3/pyvene-logo.png
.. |image2| image:: https://img.shields.io/pepy/dt/pyvene?color=green
   :target: https://pypi.org/project/pyvene/
.. |image3| image:: https://img.shields.io/pypi/v/pyvene?color=red
   :target: https://pypi.org/project/pyvene/
.. |image4| image:: https://img.shields.io/pypi/l/pyvene?color=blue
   :target: https://pypi.org/project/pyvene/

.. toctree::
   :maxdepth: 2
   :caption: Contents:

**pyvene** is an open-source Python library for intervening on the internal states of
PyTorch models. Interventions are an important operation in many areas of AI, including
model editing, steering, robustness, and interpretability.

pyvene has many features that make interventions easy:

* Interventions are the basic primitive, specified as dicts and thus able to be saved locally
  and shared as serialisable objects through HuggingFace.
* Interventions can be composed and customised: you can run them on multiple locations, on arbitrary
  sets of neurons (or other levels of granularity), in parallel or in sequence, on decoding steps of
  generative language models, etc.
* Interventions work out-of-the-box on any PyTorch model! No need to define new model classes from
  scratch and easy interventions are possible all kinds of architectures (RNNs, ResNets, CNNs, Mamba).

pyvene is under active development and constantly being improved 🫡


Getting Started
---------------

To install the latest stable version of pyvene:

::
   
   pip install pyvene

Alternatively, to install a bleeding-edge version, you can clone the repo and install:

::
   
   git clone git@github.com:stanfordnlp/pyvene.git
   cd pyvene
   pip install -e .

When you want to update, you can just run ``git pull`` in the cloned directory.

We suggest importing the library as:

.. code:: python

   import pyvene as pv


*Wrap* and *intervene*
----------------------

The usual workflow for using pyvene is to load a model, define an intervention config and wrap the model,
and then run the intervened model. This returns both the original and intervened outputs, as well as any
internal activations you specified to collect. For example:

.. code:: python

   import torch
   import pyvene as pv
   from transformers import AutoTokenizer, AutoModelForCausalLM


   # 1. Load the model
   model_name = "meta-llama/Llama-2-7b-hf" # the HF model you want to intervene on
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(
      model_name, torch_dtype=torch.bfloat16, device_map="cuda")


   # 2. Wrap the model with an intervention config
   pv_model = pv.IntervenableModel({
      "component": "model.layers[15].mlp.output",   # where to intervene (here, the MLP output in layer 15)
      "intervention": pv.ZeroIntervention           # what intervention to apply (here, zeroing out the activation)
   }, model=model)


   # 3. Run the intervened model
   orig_outputs, intervened_outputs = pv_model(
      tokenizer("The capital of Spain is", return_tensors="pt").to('cuda'),
      output_original_output=True
   )


   # 4. Compare outputs
   print(intervened_outputs.logits - orig_outputs.logits)


which returns

::

   tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.4375,  1.0625,  0.3750,  ..., -0.1562,  0.4844,  0.2969],
            [ 0.0938,  0.1250,  0.1875,  ...,  0.2031,  0.0625,  0.2188],
            [ 0.0000, -0.0625, -0.0312,  ...,  0.0000,  0.0000, -0.0156]]],
         device='cuda:0')

*Share* and *load* from HuggingFace
-----------------------------------

pyvene has support for sharing and loading intervention schemata via HuggingFace.

The following codeblock can reproduce `honest_llama-2
chat <https://github.com/likenneth/honest_llama/tree/master>`__ from the
paper `Inference-Time Intervention: Eliciting Truthful Answers from a
Language Model <https://arxiv.org/abs/2306.03341>`__. The added
activations are only **~0.14MB** on disk!

.. code:: python

   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM
   import pyvene as pv


   # 1. Load base model
   tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-2-7b-chat-hf",
       torch_dtype=torch.bfloat16,
   ).to("cuda")


   # 2. Load intervention from HF and wrap model
   pv_model = pv.IntervenableModel.load(
       "zhengxuanzenwu/intervenable_honest_llama2_chat_7B", # the activation diff ~0.14MB
       model,
   )


   # 3. Let's run it!
   print("llama-2-chat loaded with interventions:")
   q = "What's a cure for insomnia that always works?"
   prompt = tokenizer(q, return_tensors="pt").to("cuda")
   _, iti_response_shared = pv_model.generate(prompt, max_new_tokens=64, do_sample=False)
   print(tokenizer.decode(iti_response_shared[0], skip_special_tokens=True))

With this, once you discover some clever intervention schemes, you can
share with others quickly without sharing the actual base LMs or the
intervention code!

.. _intervenablemodel-as-regular-nnmodule:

*IntervenableModel* is just an *nn.Module*
------------------------------------------

pyvene wraps PyTorch models in the :class:`IntervenableModel <pyvene.models.intervenable_base.IntervenableModel>` class. This is just a subclass of
``nn.Module``, so you can use it just like any other PyTorch model! For example:

.. code:: python

   import torch
   import torch.nn as nn
   from typing import List, Optional, Tuple, Union, Dict

   class ModelWithIntervenables(nn.Module):
       def __init__(self):
           super(ModelWithIntervenables, self).__init__()
           self.pv_gpt2 = pv_gpt2
           self.relu = nn.ReLU()
           self.fc = nn.Linear(768, 1)
           # Your other downstream components go here

       def forward(
           self, 
           base,
           sources: Optional[List] = None,
           unit_locations: Optional[Dict] = None,
           activations_sources: Optional[Dict] = None,
           subspaces: Optional[List] = None,
       ):
           _, counterfactual_x = self.pv_gpt2(
               base,
               sources,
               unit_locations,
               activations_sources,
               subspaces
           )
           return self.fc(self.relu(counterfactual_x.last_hidden_state))

Complex *Intervention Schema* as an *Object*
--------------------------------------------

One key abstraction that pyvene provides is the encapsulation of the
intervention schema. While abstraction provides good user-interfaces,
pyvene can support relatively complex intervention schema. The
following helper function generates the schema for *path
patching* for individual attention heads, for replicating experiments
from the paper `Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small <https://arxiv.org/abs/2211.00593>`__:

.. code:: python

   import pyvene as pv

   def path_patching_config(
       layer, last_layer, 
       component="head_attention_value_output", unit="h.pos", 
   ):
       intervening_component = [
           {"layer": layer, "component": component, "unit": unit, "group_key": 0}]
       restoring_components = []
       if not stream.startswith("mlp_"):
           restoring_components += [
               {"layer": layer, "component": "mlp_output", "group_key": 1}]
       for i in range(layer+1, last_layer):
           restoring_components += [
               {"layer": i, "component": "attention_output", "group_key": 1}
               {"layer": i, "component": "mlp_output", "group_key": 1}
           ]
       intervenable_config = IntervenableConfig(intervening_component + restoring_components)
       return intervenable_config

You can now use the config generated by this function to wrap a model. And
after you have done your intervention, you can share your path patching
config with others:

.. code:: python

   _, tokenizer, gpt2 = pv.create_gpt2()

   pv_gpt2 = pv.IntervenableModel(
       path_patching_config(4, gpt2.config.n_layer), 
       model=gpt2
   )
   # saving the path
   pv_gpt2.save(
       save_directory="./your_gpt2_path/"
   )
   # loading the path
   pv_gpt2 = pv.IntervenableModel.load(
       "./tmp/",
       model=gpt2)

Selected Tutorials
------------------

.. list-table:: Tutorials
   :widths: 10 20 20 50
   :header-rows: 1

   * - Level
     - Tutorial
     - Run in Colab
     - Description
   * - Beginner
     - `pyvene 101 <tutorials/pyvene_101.html>`__
     - 
         .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :align: center
            :target: https://colab.research.google.com/github/stanfordnlp/pyvene/blob/main/pyvene_101.ipynb
     - Introduce you to the basics of pyvene
   * - Intermediate
     - `ROME Causal Tracing <tutorials/advanced_tutorials/Causal_Tracing.html>`__
     - 
         .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :align: center
            :target: https://colab.research.google.com/github/stanfordnlp/pyvene/blob/main/tutorials/advanced_tutorials/Causal_Tracing.ipynb
     - Reproduce ROME's Results on Factual Associations with GPT2-XL
   * - Intermediate
     - `Intervention vs. Probing <tutorials/advanced_tutorials/Probing_Gender.html>`__
     - 
         .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :align: center
            :target: https://colab.research.google.com/github/stanfordnlp/pyvene/blob/main/tutorials/advanced_tutorials/Probing_Gender.ipynb
     - Illustrates how to run trainable interventions and probing with pythia-6.9B
   * - Advanced
     - `Trainable Interventions for Causal Abstraction <tutorials/advanced_tutorials/DAS_Main_Introduction.html>`__
     - 
         .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :align: center
            :target: https://colab.research.google.com/github/stanfordnlp/pyvene/blob/main/tutorials/advanced_tutorials/DAS_Main_Introduction.ipynb
     - Illustrates how to train an intervention to discover causal mechanisms of a neural model
      

Contributing to This Library
----------------------------

Please see `our guidelines <guides/contributing.html>`__ about how to contribute
to this repository.

*Pull requests, bug reports, and all other forms of contribution are
welcomed and highly encouraged!*

Citation
--------

If you use this repository, please cite our library paper:

.. code:: bibtex

   @inproceedings{wu-etal-2024-pyvene,
      title = "pyvene: A Library for Understanding and Improving {P}y{T}orch Models via Interventions",
      author = "Wu, Zhengxuan and Geiger, Atticus and Arora, Aryaman and Huang, Jing and Wang, Zheng and Goodman, Noah and Manning, Christopher and Potts, Christopher",
      editor = "Chang, Kai-Wei and Lee, Annie and Rajani, Nazneen",
      booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: System Demonstrations)",
      month = jun,
      year = "2024",
      address = "Mexico City, Mexico",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2024.naacl-demo.16",
      pages = "158--165",
   }


Related Works on Discovering Causal Mechanism of LLMs
-----------------------------------------------------

If you would like to read more works on this area, here is a list of
papers that try to align or discover the causal mechanisms of LLMs.

-  `Causal Abstractions of Neural
   Networks <https://arxiv.org/abs/2106.02997>`__: This paper introduces
   interchange intervention (a.k.a. activation patching or causal
   scrubbing). It tries to align a causal model with the model's
   representations.
-  `Inducing Causal Structure for Interpretable Neural
   Networks <https://arxiv.org/abs/2112.00826>`__: Interchange
   intervention training (IIT) induces causal structures into the
   model's representations.
-  `Localizing Model Behavior with Path
   Patching <https://arxiv.org/abs/2304.05969>`__: Path patching (or
   causal scrubbing) to uncover causal paths in neural model.
-  `Towards Automated Circuit Discovery for Mechanistic
   Interpretability <https://arxiv.org/abs/2304.14997>`__: Scalable
   method to prune out a small set of connections in a neural network
   that can still complete a task.
-  `Interpretability in the Wild: a Circuit for Indirect Object
   Identification in GPT-2 small <https://arxiv.org/abs/2211.00593>`__:
   Path patching plus posthoc representation study to uncover a circuit
   that solves the indirect object identification (IOI) task.
-  `Rigorously Assessing Natural Language Explanations of
   Neurons <https://arxiv.org/abs/2309.10312>`__: Using causal
   abstraction to validate `neuron explanations released by
   OpenAI <https://openai.com/research/language-models-can-explain-neurons-in-language-models>`__.

Star History
------------

|Star History Chart|

.. |Star History Chart| image:: https://api.star-history.com/svg?repos=stanfordnlp/pyvene,stanfordnlp/pyreft&type=Date
   :target: https://star-history.com/#stanfordnlp/pyvene&stanfordnlp/pyreft&Date

.. toctree::
   :hidden:
   :caption: Guides

   guides/contributing
   guides/causal_abstraction
   guides/ndif


.. toctree::
   :hidden:
   :caption: Basic tutorials

   tutorials/pyvene_101
   tutorials/basic_tutorials/Add_Activations_to_Streams
   tutorials/basic_tutorials/Basic_Intervention
   tutorials/basic_tutorials/Nested_Intervention
   tutorials/basic_tutorials/Subspace_Partition_with_Intervention
   tutorials/basic_tutorials/Intervention_Training

.. toctree::
   :hidden:
   :caption: Advanced tutorials

   tutorials/advanced_tutorials/Causal_Tracing
   tutorials/advanced_tutorials/Probing_Gender
   tutorials/advanced_tutorials/DAS_Main_Introduction
   tutorials/advanced_tutorials/Boundless_DAS
   tutorials/advanced_tutorials/IOI_Replication
   tutorials/advanced_tutorials/IOI_with_DAS
   tutorials/advanced_tutorials/IOI_with_Mask_Intervention
   tutorials/advanced_tutorials/Interventions_with_BLIP
   tutorials/advanced_tutorials/MQNLI
   tutorials/advanced_tutorials/Voting_Mechanism

.. toctree::
   :hidden:
   :caption: API

   api/core


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`