.. pyvene documentation master file, created by
   sphinx-quickstart on Fri Jul 12 16:49:16 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. container::

   .. rubric:: |image1|
      :name: section

   `Read Our Paper » <https://arxiv.org/abs/2403.07809>`__

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


pyvene
======

Interventions on model-internal states are fundamental operations in many areas of AI,
including model editing, steering, robustness, and interpretability. To facilitate such research,
we introduce **pyvene**, an open-source Python library that supports customizable interventions
on a range of different PyTorch modules. pyvene supports complex intervention schemes with
an intuitive configuration format, and its interventions can be static or include trainable
parameters.

Getting Started
---------------

Since the library is evolving, it is recommended to install pyvene by,

::
   
   git clone git@github.com:stanfordnlp/pyvene.git

and add pyvene into your system path in Python via,

::

   import sys
   sys.path.append("<Your Path to Pyvene>")

   import pyvene as pv

Alternatively, you can do

::

   pip install git+https://github.com/stanfordnlp/pyvene.git

or

::

   pip install pyvene


*Wrap*, *intervene*, and *share*
--------------------------------

You can intervene with any HuggingFace model as,

::

   import torch
   import pyvene as pv
   from transformers import AutoTokenizer, AutoModelForCausalLM

   model_name = "meta-llama/Llama-2-7b-hf" # your HF model name.
   model = AutoModelForCausalLM.from_pretrained(
      model_name, torch_dtype=torch.bfloat16, device_map="cuda")
   tokenizer = AutoTokenizer.from_pretrained(model_name)

   def zeroout_intervention_fn(b, s): 
      b[:,3] = 0. # 3rd position
      return b

   pv_model = pv.IntervenableModel({
      "component": "model.layers[15].mlp.output", # string access
      "intervention": zeroout_intervention_fn}, model=model)

   # run the intervened forward pass
   orig_outputs, intervened_outputs = pv_model(
      tokenizer("The capital of Spain is", return_tensors="pt").to('cuda'),
      output_original_output=True
   )
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

*IntervenableModel* Loaded from HuggingFace Directly
----------------------------------------------------

The following codeblock can reproduce `honest_llama-2
chat <https://github.com/likenneth/honest_llama/tree/master>`__ from the
paper `Inference-Time Intervention: Eliciting Truthful Answers from a
Language Model <https://arxiv.org/abs/2306.03341>`__. The added
activations are only **~0.14MB** on disk!

.. code:: python

   # others can download from huggingface and use it directly
   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM
   import pyvene as pv

   tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-2-7b-chat-hf",
       torch_dtype=torch.bfloat16,
   ).to("cuda")

   pv_model = pv.IntervenableModel.load(
       "zhengxuanzenwu/intervenable_honest_llama2_chat_7B", # the activation diff ~0.14MB
       model,
   )

   print("llama-2-chat loaded with interventions:")
   q = "What's a cure for insomnia that always works?"
   prompt = tokenizer(q, return_tensors="pt").to("cuda")
   _, iti_response_shared = pv_model.generate(prompt, max_new_tokens=64, do_sample=False)
   print(tokenizer.decode(iti_response_shared[0], skip_special_tokens=True))

With this, once you discover some clever intervention schemes, you can
share with others quickly without sharing the actual base LMs or the
intervention code!

.. _intervenablemodel-as-regular-nnmodule:

*IntervenableModel* as Regular *nn.Module*
------------------------------------------

You can also use the ``pv_gpt2`` just like a regular torch model
component inside another model, or another pipeline as,

.. code:: py

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

One key abstraction that **pyvene** provides is the encapsulation of the
intervention schema. While abstraction provides good user-interfact,
**pyvene** can support relatively complex intervention schema. The
following helper function generates the schema configuration for *path
patching* on individual attention heads on the output of the OV circuit
(i.e., analyzing causal effect of each individual component):

.. code:: py

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

then you can wrap the config generated by this function to a model. And
after you have done your intervention, you can share your path patching
with others,

.. code:: py

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
     - `pyvene 101 <pyvene_101.ipynb>`__
     - 
         .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :align: center
            :target: https://colab.research.google.com/github/stanfordnlp/pyvene/blob/main/pyvene_101.ipynb
     - Introduce you to the basics of pyvene
   * - Intermediate
     - `ROME Causal Tracing <tutorials/advanced_tutorials/Causal_Tracing.ipynb>`__
     - 
         .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :align: center
            :target: https://colab.research.google.com/github/stanfordnlp/pyvene/blob/main/tutorials/advanced_tutorials/Causal_Tracing.ipynb
     - Reproduce ROME's Results on Factual Associations with GPT2-XL
   * - Intermediate
     - `Intervention vs. Probing <tutorials/advanced_tutorials/Probing_Gender.ipynb>`__
     - 
         .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :align: center
            :target: https://colab.research.google.com/github/stanfordnlp/pyvene/blob/main/tutorials/advanced_tutorials/Probing_Gender.ipynb
     - Illustrates how to run trainable interventions and probing with pythia-6.9B
   * - Advanced
     - `Trainable Interventions for Causal Abstraction <tutorials/advanced_tutorials/DAS_Main_Introduction.ipynb>`__
     - 
         .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :align: center
            :target: https://colab.research.google.com/github/stanfordnlp/pyvene/blob/main/tutorials/advanced_tutorials/DAS_Main_Introduction.ipynb
     - Illustrates how to train an intervention to discover causal mechanisms of a neural model
      

Contributing to This Library
----------------------------

Please see `our guidelines <CONTRIBUTING.md>`__ about how to contribute
to this repository.

*Pull requests, bug reports, and all other forms of contribution are
welcomed and highly encouraged!* :octocat:

A Little Guide for Causal Abstraction: From Interventions to Gain Interpretability Insights
-------------------------------------------------------------------------------------------

Basic interventions are fun but we cannot make any causal claim
systematically. To gain actual interpretability insights, we want to
measure the counterfactual behaviors of a model in a data-driven
fashion. In other words, if the model responds systematically to your
interventions, then you start to associate certain regions in the
network with a high-level concept. We also call this alignment search
process with model internals.

Understanding Causal Mechanisms with Static Interventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is a more concrete example,

.. code:: py

   def add_three_numbers(a, b, c):
       var_x = a + b
       return var_x + c

The function solves a 3-digit sum problem. Let's say, we trained a
neural network to solve this problem perfectly. "Can we find the
representation of (a + b) in the neural network?". We can use this
library to answer this question. Specifically, we can do the following,

-  **Step 1:** Form Interpretability (Alignment) Hypothesis: We
   hypothesize that a set of neurons N aligns with (a + b).
-  **Step 2:** Counterfactual Testings: If our hypothesis is correct,
   then swapping neurons N between examples would give us expected
   counterfactual behaviors. For instance, the values of N for (1+2)+3,
   when swapping with N for (2+3)+4, the output should be (2+3)+3 or
   (1+2)+4 depending on the direction of the swap.
-  **Step 3:** Reject Sampling of Hypothesis: Running tests multiple
   times and aggregating statistics in terms of counterfactual behavior
   matching. Proposing a new hypothesis based on the results.

To translate the above steps into API calls with the library, it will be
a single call,

.. code:: py

   intervenable.eval_alignment(
       train_dataloader=test_dataloader,
       compute_metrics=compute_metrics,
       inputs_collator=inputs_collator
   )

where you provide testing data (basically interventional data and the
counterfactual behavior you are looking for) along with your metrics
functions. The library will try to evaluate the alignment with the
intervention you specified in the config.

--------------

Understanding Causal Mechanism with Trainable Interventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The alignment searching process outlined above can be tedious when your
neural network is large. For a single hypothesized alignment, you
basically need to set up different intervention configs targeting
different layers and positions to verify your hypothesis. Instead of
doing this brute-force search process, you can turn it into an
optimization problem which also has other benefits such as distributed
alignments.

In its crux, we basically want to train an intervention to have our
desired counterfactual behaviors in mind. And if we can indeed train
such interventions, we claim that causally informative information
should live in the intervening representations! Below, we show one type
of trainable intervention
``models.interventions.RotatedSpaceIntervention`` as,

.. code:: py

   class RotatedSpaceIntervention(TrainableIntervention):
       
       """Intervention in the rotated space."""
       def forward(self, base, source):
           rotated_base = self.rotate_layer(base)
           rotated_source = self.rotate_layer(source)
           # interchange
           rotated_base[:self.interchange_dim] = rotated_source[:self.interchange_dim]
           # inverse base
           output = torch.matmul(rotated_base, self.rotate_layer.weight.T)
           return output

Instead of activation swapping in the original representation space, we
first **rotate** them, and then do the swap followed by un-rotating the
intervened representation. Additionally, we try to use SGD to **learn a
rotation** that lets us produce expected counterfactual behavior. If we
can find such rotation, we claim there is an alignment.
``If the cost is between X and Y.ipynb`` tutorial covers this with an
advanced version of distributed alignment search, `Boundless
DAS <https://arxiv.org/abs/2305.08809>`__. There are `recent
works <https://www.lesswrong.com/posts/RFtkRXHebkwxygDe2/an-interpretability-illusion-for-activation-patching-of>`__
outlining potential limitations of doing a distributed alignment search
as well.

You can now also make a single API call to train your intervention,

.. code:: py

   intervenable.train_alignment(
       train_dataloader=train_dataloader,
       compute_loss=compute_loss,
       compute_metrics=compute_metrics,
       inputs_collator=inputs_collator
   )

where you need to pass in a trainable dataset, and your customized loss
and metrics function. The trainable interventions can later be saved on
to your disk. You can also use ``intervenable.evaluate()`` your
interventions in terms of customized objectives.

Citation
--------

If you use this repository, please consider to cite our library paper:

.. code:: stex

   @article{wu2024pyvene,
     title={pyvene: A Library for Understanding and Improving {P}y{T}orch Models via Interventions},
     author={Wu, Zhengxuan and Geiger, Atticus and Arora, Aryaman and Huang, Jing and Wang, Zheng and Noah D. Goodman and Christopher D. Manning and Christopher Potts},
     booktitle={arXiv:2403.07809},
     url={arxiv.org/abs/2403.07809},
     year={2024}
   }

Related Works in Discovering Causal Mechanism of LLMs
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