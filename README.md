<br />
<div align="center">
  <h1 align="center"><img src="https://i.ibb.co/N1kYZy5/icon.png" width="30" height="30"> align-transformers</h1>
  <a href="https://arxiv.org/abs/2305.08809"><strong>Read Our Recent Paper »</strong></a>
</div>


## <img src="https://i.ibb.co/N1kYZy5/icon.png" width="30" height="30"> **Finding Alignment of Model Internals with Customizable Interventions**
We release this **new** generic library for studying model internals encapsulating **causal abstraction and distributed alignment search**[^ii], **path patching**[^pp], **causal scrubbing**[^cs] introduced recently for finding causal alignments with the internals of neural models. This library is designed as a playground for inventing new interventions, whether trainable or not, to discover the causal mechanisms of neural models. This library also focuses on scaling these methods to LLMs with billions of parameters.


## Release Notes
:white_check_mark: 05/17/2023 - Preprint with the initial version of align-transformers is released! Read this for a more formal definition of the method.      
:white_check_mark: 10/04/2023 - Major infrastructure change to hook-based and customizable interventions. To reproduce old experiments [our NeurIPS 2023 paper](https://arxiv.org/abs/2305.08809), please use our shelved release version [here](https://github.com/frankaging/align-transformers/releases/tag/NeurIPS-2023). For this repository, we focus on the generic alignment library from now on.    


## How to Intervene?
We design this library to be flexible and extensible to all kinds of interventions, causal mechanism alignments, and model types. The basic idea is we sample representations that we want to align from different training examples, do representation interventions, and study the model's behavior changes. The intervention can be trainable (i.e., DAS) or static (i.e., causal scrubbing).

#### Loading models from huggingface
```py
from models.utils import create_gpt2

config, tokenizer, gpt = create_gpt2()
```

#### Create a simple alignable config
```py
alignable_config = AlignableConfig(
    alignable_model_type="gpt2",
    alignable_representations=[
        AlignableRepresentationConfig(
            0,            # intervening layer 0
            "mlp_output", # intervening mlp output
            "pos",        # intervening based on positional indices of tokens
            1             # maximally intervening one token
        ),
    ],
)
```

#### Turn the model into an alignable object
The basic idea is to think alignable model as a regular huggingface model except it supports an intervenable forward function instead.
```py
alignable_gpt = AlignableModel(alignable_config, gpt)
```

#### Intervene with examples
```py
base = tokenizer("The capital of Spain is", return_tensors="pt")
sources = [tokenizer("The capital of Italy is", return_tensors="pt")]

_, counterfactual_outputs = alignable_gpt(
    base,
    sources,
    {"sources->base": ([[[4]]], [[[4]]])} # intervene base with sources
)
```


## A Set of Tutorials
We will release a set of tutorials focusing on different methods like **causal abstraction and distributed alignment search**[^ii], **path patching**[^pp], **causal scrubbing**[^cs] using this library. 

### `The capital of Spain is.ipynb` 
This is a tutorial for doing simple path patching as in **Path Patching**[^pp], **Causal Scrubbing**[^cs]. Thanks to [Aryaman Arora](https://aryaman.io/). This is a set of experiments trying to reproduce some of the experiments in his awesome [nano-causal-interventions](https://github.com/aryamanarora/nano-causal-interventions) repository.

### `If the cost is between X and Y.ipynb` 
This is a tutorial reproducing one of the main experiments in [the Boundless DAS paper](https://arxiv.org/abs/2305.08809). Different from the first tutorial, this one involves trainable interventions that actively search for alignments with model internals.

### `Hook with new model and intervention types.ipynb` 
This is a tutorial on integrating new model types with this library as well as customized interventions. We try to add `flan_t5` as well as a simple additive intervention. This tutorial covers a simple experiment as in `The capital of Spain is.ipynb`.


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


## Citation
If you use this repository, please consider to cite relevant papers from our group:
```stex
  @article{geiger-etal-2023-DAS,
        title={Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations}, 
        author={Geiger, Atticus and Wu, Zhengxuan and Potts, Christopher and Icard, Thomas  and Goodman, Noah},
        year={2023},
        booktitle={arXiv}
  }

  @article{wu-etal-2023-Boundless-DAS,
        title={Interpretability at Scale: Identifying Causal Mechanisms in Alpaca}, 
        author={Wu, Zhengxuan and Geiger, Atticus and Icard, Thomas and Potts, Christopher and Goodman, Noah},
        year={2023},
        booktitle={NeurIPS}
  }
```

[^pp]: [Wang et al. (2022)](https://arxiv.org/abs/2211.00593), [Goldowsky-Dill et al. (2023)](https://arxiv.org/abs/2304.05969)
[^cs]: [Chan et al. (2022)](https://www.lesswrong.com/s/h95ayYYwMebGEYN5y)
[^ii]: [Geiger et al. (2021a)](https://arxiv.org/abs/2106.02997), [Geiger et al. (2021b)](https://arxiv.org/abs/2112.00826) (but I only consider inference-time intervention), [Geiger et al. (2023)](https://arxiv.org/abs/2301.04709), [Wu et al. (2023)](https://arxiv.org/pdf/2303.02536)
