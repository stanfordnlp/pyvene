<br />
<div align="center">
  <h3 align="center"><img src="https://i.ibb.co/N1kYZy5/icon.png" width="30" height="30"> Interpretability at Scale: Identifying Causal Mechanisms in Alpaca</h3>
  <p align="center">
    Zhengxuan Wu*, Atticus Geiger*, Christopher Potts, Noah Goodman
    <br />
    <a href="https://arxiv.org/abs/xxxx.xxxxx"><strong>Read our preprint »</strong></a>
    <br />
    <br />
    <a href="https://nlp.stanford.edu/~wuzhengx/boundless_das/">Research Blog</a>
    ·
    <a href="https://nlp.stanford.edu/~wuzhengx/boundless_das/cn_index.html">中文介绍</a>
    ·
    <a href="https://github.com/frankaging/align-transformers/issues">Report Bug</a>
    ·
    <a href="https://nlp.stanford.edu/~wuzhengx/">Contact Us</a>
  </p>
</div>

## <img src="https://i.ibb.co/N1kYZy5/icon.png" width="30" height="30"> **align-transformers**
Obtaining robust, human-interpretable explanations of large, general-purpose language models is an urgent goal for AI. Building on the theory of causal abstraction, we release this generic  library encapsulated Boundless DAS introduced in our paper for find representations that play a given causal role in LLMs with billions of parameters.

## End-to-end Alignment Workflow for Any LLMs.
- **Step 1** : Identify a reasoning task that can be solved by a symbolic causal model.
- **Step 2** : Pick an off-the-shelf LLM.
- **Step 3** : Make sure LLM passes the behavioral test (i.e., good task performance)
- **Step 4** : Iteratively align the LLM with the causal model using Boundless DAS.

## Release Notes
:white_check_mark: 05/17/2023 - Preprint with the initial version of align-transformers is released! Read this for a more formal definition of the method.   
:white_check_mark: 05/17/2023 - Support LLaMA model with a simple reasoning task.  
⬜️ Support LLaMA model (>30B) training with model sharding.  
⬜️ Support other models.

## Codebase Structure and How to Contribute
```.
├── models
│   ├── llama
│   │   ├── modelings_alignable_llama.py
│   └── configuration_alignable_model.py
│
├── notebooks
│   ├── analysis.ipynb
│   ├── check_finished_experiments.ipynb
│   └── cevaluation.ipynb
│
├── torch3.8_overwrite
│   ├── init.py
│   └── cevaluation.ipynb
│ 
├── counterfacutal_datasets.py
├── trainer.py
└── run_alignment.py
 ```
 We follow huggingface transformers library closely to organize our folder. To contribute or adapt this codebase for your own analyses, here are some pointers:
 - **New Models** : Follow the `modelings_alignable_llama.py` to create your own model file just like transformers ones. Typically, you only need to add < 50 lines of code to make it work.
 - **New Dataset / Task** : Follow the `counterfacutal_datasets.py` file to create your own dataset. The training datset is encapsulated using huggingface Datasets object. Here is one example:
```python
train_dataset = Dataset.from_dict(
    {
        "input_ids": raw_train[0], 
        "source_input_ids": raw_train[1],
        "labels": raw_train[2],
        "intervention_ids": raw_train[3],
    }
).with_format("torch")
```
Any dataset instance following the format above should automatically work with the current trainer code.

## Citation
If you use this repository, please consider to cite our relevant papers:
```stex
  @article{wu-etal-2023-Boundless-DAS,
        title={Interpretability at Scale: Identifying Causal Mechanisms in Alpaca}, 
        author={Wu, Zhengxuan and Geiger, Atticus and Potts, Christopher and Goodman, Noah},
        year={2023},
        eprint={xxxx.xxxxx},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
  }
```

## System Requirements
- Python 3.8 are supported.
- Pytorch Version: 1.13.1
- Transfermers Minimum Version: 4.28.0.dev0
- Datasets Version: Version: 2.3.2

### `bf16` training with customized pytorch files
To save memory and speed up the training, we allow `bf16` mode when finding alignments. Since we rely on `torch` orthogonalization library, it does not support `bf16`. So, we did some hacks in the `torch` file to enable this. Modified files are in the `torch3.8_overwrite/*.py` folder. Currently, you need to replace these two files by hand in your environment. Here are two example directories for these two files:
```
/lib/python3.8/site-packages/torch/nn/utils/parametrizations.py
/lib/python3.8/site-packages/torch/nn/init.py
```


## Model Requirements
LLMs raw weights are not provided in this repository. Please download the model weights separately. And our codebase should work fine with any model that is saved as huggingface transformers format (e.g., saved using `save_pretrained(YOUR_DIRECTORY)`). The external model folder should look like this,
```.
├── das_config
│   └── config.json
│
├── added_tokens.json
├── config.json
├── pytorch_model.bin
├── special_tokens_map.json
├── tokenizer.model
└── tokenizer_config.json
 ```

In the model folder, you also need to provide a separate config file as in `das_config/config.json` for Boundless DAS like this one,
```json
{
  "das_layer": 15,
  "das_token_range": [
    80,
    81
  ],
  "model_type": "llama",
  "transformers_version": "4.28.0.dev0"
}
```
Here, we tell the alignment trainer which layer and what position to look for alignment.

## Training for Boundless DAS
Here is an example of how to run training script,
```bash
python run_alignment.py \
--model_path ../alpaca_test \
--train_batch_size 16 \
--eval_batch_size 16 \
--gradient_accumulation_steps 4 \
--lr 1e-3 \
--seed 42 \
--output_dir ./results_test/ \
--epochs 3 \
--do_align \
--n_training_examples 20000 \
--n_eval_examples 1000 \
--task_name pricing_tag_lub \
--bf16
```
You can use `--bf16` to use the bfloat16 for faster training with minimum drops in percision.

