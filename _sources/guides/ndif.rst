NDIF Integration
================

We are working with the `NDIF <https://ndif.us/>`_ team to support remote intervention calls
without asking the users to download or host their own LLMs! This is still under construction.

First of all, you need to install ``nnsight``:

::

   pip install nnsight

All you have to do is to use NDIF library to load your model and use pyvene to wrap it
(i.e., pyvene will automatically recognize NDIF models)! Here is an example:

::

   from nnsight import LanguageModel
   
   # load nnsight.LanguageModel as your model to wrap with pyvene
   gpt2_ndif = LanguageModel('openai-community/gpt2', device_map='cpu')
   tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

   # pyvene provides pv.build_intervenable_model as the generic model builder
   pv_gpt2_ndif = pv.build_intervenable_model({
      "component": "transformer.h[10].attn.attn_dropout.input",
      "intervention": pv.CollectIntervention()}, model=gpt2_ndif, remote=False)


Then, you can use ``pv_gpt2_ndif`` as your regular intervenable model.
If you specify ``remote=True`` (this is still under construction), then
everything will be executed remotely on NDIF server with **zero** GPU
resource required! We provide example code in our `main tutorial <tutorials/pyvene_101>`__.