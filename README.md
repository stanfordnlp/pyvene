<br />
<div align="center">
  <h3 align="center"><img src="https://i.ibb.co/N1kYZy5/icon.png" width="30" height="30"> Interpretability at Scale: Identifying Causal Mechanisms in Alpaca</h3>
  <p align="center">
    Zhengxuan Wu*, Atticus Geiger*, Christopher Potts, Noah Goodman
    <br />
    <a href="https://github.com/frankaging/align-transformers/blob/main/tutorial.ipynb"><strong>Tutorial »</strong></a>
    <br />
    <a href="https://docs.google.com/document/d/1KDHfow4AWfHo9XxmkhCNSME9xdOdvQogGw4gbcwTKPU/edit?usp=sharing"><strong>Runbook »</strong></a>
    <br />
    <a href="https://arxiv.org/abs/2305.08809"><strong>Read our preprint »</strong></a>
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
Obtaining robust, human-interpretable explanations of large, general-purpose language models is an urgent goal for AI. Building on the theory of causal abstraction, we release this generic  library encapsulating Boundless DAS introduced in our paper for finding representations that play a given causal role in LLMs with billions of parameters.

## A Step-by-step Runbook (from scratch using AWS cloud)
We now have a detailed runbook for setting the environment for training boundless DAS from scratch using an EC2 instance from AWS cloud. You can find [the runbook](https://docs.google.com/document/d/1KDHfow4AWfHo9XxmkhCNSME9xdOdvQogGw4gbcwTKPU/edit?usp=sharing) here. You are very welcomed to contribute by making comments on the tutorial document. We will update accordingly and put your name on it.

## Our `tutorial.ipynb` Is All Your Need
Since the release of the paper, we got requests about making a onboarding tutorial of boundless DAS. Now, it is in `tutorial.ipynb`. It contains steps needed to reproduce results in our paper. **Additionally, it contains many extra fun stuff that are not discussed in the paper: federated model steering and community building!** We really hope this project can contribute to a very interesting and new topic **federated model steering** where we steer model's behavior through causal lens at inference time in a federated way using representation-based intervention.


## Release Notes
:white_check_mark: 05/17/2023 - Preprint with the initial version of align-transformers is released! Read this for a more formal definition of the method.   
:white_check_mark: 05/17/2023 - Support LLaMA model with a simple reasoning task.  
:white_check_mark: 05/31/2023 - Infra updates to decouple trainer, metrics, model loading, dataset loader; Support GPT2 alignment. Initialize the example folder for 
analyzing finetuned models.   
:white_check_mark: 06/27/2023 - A full tutorial notebook `tutorial.ipynb` with [a runbook](https://docs.google.com/document/d/1KDHfow4AWfHo9XxmkhCNSME9xdOdvQogGw4gbcwTKPU/edit?usp=sharing) on boundless DAS.    
:white_check_mark: 07/20/2023 - Another tutorial on interpreting causal graphs in a word logic puzzle with GPT-2 model in `examples/logic_game/tutorial.ipynb`! We explain the problem and alignment in details, and *we are still on our way to uncover the causal graph*. If you are interested in this problem, feel free to contact us!   
:white_check_mark: 08/09/2023 - **Rotation Learning Update** We add into GPT-2 alignable model an additional loss to prefer equally good permutation matrix. Explanations can be found `localist_paradox.ipynb`!   
⬜️ Support LLaMA model (>30B) training with model sharding (Soon! with instruction-tuning steps to make sure we are using a good template). 

## Codebase Structure and How to Contribute
```.
├── models
│   ├── llama
│   │   └── modelings_alignable_llama.py
│   ├── gpt2
│   │   └── modelings_alignable_gpt2.py
│   ├── ...
│   │   └── modelings_alignable_*.py
│   │
│   ├── configuration_alignable_model.py
│   └── modelings_alignable.py
│
├── counterfacutal_datasets
│   ├── price_tagging_game.py
│   └── *.py
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
├── examples
│   └── *.py
│ 
├── requirement.txt
├── tutorial.ipynb
├── trainer.py
└── run_alignment.py
 ```
 We follow huggingface transformers library closely to organize our folder. To contribute or adapt this codebase for your own analyses, here are some pointers:
 - **New Models** : Follow the `modelings_alignable_llama.py` to create your own model file just like transformers ones. Typically, you only need to add < 50 lines of code to make it work.
 - **New Dataset / Task** : Follow files in `counterfacutal_datasets` to create your own dataset. The training datset is encapsulated using huggingface Datasets object. Here is one example:
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

### Examples Folder
For cases where we need to *train* a model before alignment, we provide some examples coming off from models we trained to solve some reasoning puzzles. Normally, the tasks we are looking at are reasoning tasks that involve multi-step reasonings. In the alignment process, we will then try to see if the model (i.e., the neural network) is solving a task like a human task taker.

## Citation
If you use this repository, please consider to cite our relevant papers:
```stex
  @article{wu-etal-2023-Boundless-DAS,
        title={Interpretability at Scale: Identifying Causal Mechanisms in Alpaca}, 
        author={Wu, Zhengxuan and Geiger, Atticus and Potts, Christopher and Goodman, Noah},
        year={2023},
        eprint={2305.08809},
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

