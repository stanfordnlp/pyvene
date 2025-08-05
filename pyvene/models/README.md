## How to add new models?

You can prompt a LM to generate files, or modifying existing ones in this folder by simply following these steps:

- Get the relevent implementation file from `https://github.com/huggingface/transformers/blob/main/src/transformers/models/` (e.g., the implementation for `gpt-oss` [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py)).

- Copy the whole transformer model src file.

- Create a new folder for your new model.

- Move one of the existing model file to your new folder (e.g., `/gpt2/modelings_intervenable_gpt2.py` along with the default `__init__.py` file).

- Prompt a language model with the following template:

```text
[YOUR_EXAMPLE_PYVENE_MODEL_FILE_COPY]

Generate a new mapping file based on the existing one above for the following new model:

[HF_TRANSFORMER_MODEL_SRC_FILE_COPY]

You also need to pay attention to these details:
- [OTHER_REQ_GOES_HERE] (e.g., you need to take care of the MoE strcuture)
```
