<br />
<div align="center">
  <h1 align="center"><img src="https://i.ibb.co/N1kYZy5/icon.png" width="30" height="30"> align-transformers</h1>
  <a href="https://arxiv.org/abs/2305.08809"><strong>Read Our Recent Paper »</strong></a>
</div>

[<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/frankaging/align-transformers/blob/main/tutorials/The%20capital%20of%20Spain%20is.ipynb)

# <img src="https://i.ibb.co/N1kYZy5/icon.png" width="30" height="30"> **Aligning Causal Mechanisms with Transformer Model Internals with Interventions**
We have released a **new** generic library for studying model internals, which encapsulates **causal abstraction and distributed alignment search**[^ii], **path patching**[^pp], and **causal scrubbing**[^cs]. These methods were introduced recently to find or to help us find causal alignments with the internals of neural models. This library is designed as a playground for inventing new interventions, whether they're trainable or not, to uncover the causal mechanisms of neural models. Additionally, the library emphasizes scaling these methods to LLMs with billions of parameters.


## Release Notes
:white_check_mark: 05/17/2023 - Preprint with the initial version of align-transformers is released!  
:white_check_mark: 10/04/2023 - Major infrastructure change to support hook-based and customizable interventions. To reproduce old experiments in [our NeurIPS 2023 paper](https://arxiv.org/abs/2305.08809), please use our shelved version [here](https://github.com/frankaging/align-transformers/releases/tag/NeurIPS-2023).

## Interventions, Alignments, and Distributed Alignments
In this section, we discuss topics from interventions to alignments, and more recent distributed alignments.

### Interventions
We've redesigned this library to be flexible and extensible for all types of interventions, causal mechanism alignments, and model varieties. The basic concept is to sample representations we wish to align from various training examples, perform representation interventions, and then observe changes in the model's behavior. These interventions can be trainable (e.g., DAS) or static (e.g., causal scrubbing).

#### Loading models from HuggingFace
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
The basic idea is to consider the alignable model as a regular HuggingFace model, except that it supports an intervenable forward function.
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
--- 

### Alignments
What are alignments with model internals? An alignment is defined between a high-level concept or variable and a set of low-level neurons or activations. When we say there is an alignment between them, we basically claim that, when intervening on them respectively, we cannot distinguish them (a.k.a. they have the same causal behaviors under interventions). Here is one simple example:
```py
def add_three_numbers(a, b, c, source_a_add_b=None):
    if source_a_add_b:
        return source_a_add_b + c
    return a + b + c
```
The function above forms a very simple intervenable high-level causal model for solving a 3-digit sum problem (`source_a_add_b` will take in the intermediate result of `(a + b)` from another example). Let's say, we trained a neural network to solve this problem perfectly. **One concrete alignment problem is** "Can we find the representation of (a + b) in the neural network that we trained to solve this problem?". To solve this problem, we use interventions. 

- **Step 1:** Form Alignment Hypothesis: We hypothesize that a set of neurons N aligns with (a + b).
- **Step 2:** Counterfactual Testings: If our hypothesis is correct, then swapping neurons N between examples would give us expected counterfactual behaviors. For instance, the values of N for (1+2)+3, when swapping with N for (2+3)+4, the output should be (2+3)+3 or (1+2)+4 depending on the direction of the swap.
- **Step 2:** Reject Sampling of Hypothesis: Running tests multiple times and aggregating statistics in terms of counterfactual behavior matching. Proposing a new hypothesis based on the results. 

**We will soon provide a tutorial on this.** This library supports this process, you just need to manually inspect how good the matchings are by taking a look at the counterfactual output. [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997) implements the exact steps if you want to learn more about it.

--- 

### Distributed Alignments
One key limitation of the process above is that it **assumes** the alignment is between individual neurons/activations and a high-level variable. **This is often falsifiable**. [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) explores some of the issues. Basically, one high-level variable may be aligned with a whole layer, or a single neuron may be aligned with a set of high-level variables. In other words, we need distributed alignment, moving away from localist alignment where we assume there exists an almost one-to-one mapping between a neuron and a high-level variable. 

`models.interventions.RotatedSpaceIntervention` in our codebase provides a way to do a distributed alignment search.
```py
class RotatedSpaceIntervention(TrainbleIntervention):
    
    """Intervention in the rotated space."""
    def forward(self, base, source):
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # interchange
        rotated_base[:self.interchange_dim] = rotated_source[:self.interchange_dim]
        # inverse base
        output = torch.matmul(rotated_base, self.rotate_layer.weight.T)
        return output
```
Instead of activation swapping in the original representation space, we first **rotate** them, and then do the swap followed by un-rotating the intervened representation. Additionally, we try to use SGD to **learn a rotation** that lets us produce expected counterfactual behavior. If we can find such rotation, we claim there is an alignment. `If the cost is between X and Y.ipynb` tutorial covers this with an advanced version of distributed alignment search, [Boundless DAS](https://arxiv.org/abs/2305.08809). There are [recent works](https://www.lesswrong.com/posts/RFtkRXHebkwxygDe2/an-interpretability-illusion-for-activation-patching-of) outlining potential limitations of doing a distributed alignment search as well.

## Tutorials
We released a set of tutorials for doing model interventions and model alignments.

### `The capital of Spain is.ipynb` 
(**Intervention Tutorial**) This is a tutorial for doing simple path patching as in **Path Patching**[^pp], **Causal Scrubbing**[^cs]. Thanks to [Aryaman Arora](https://aryaman.io/). This is a set of experiments trying to reproduce some of the experiments in his awesome [nano-causal-interventions](https://github.com/aryamanarora/nano-causal-interventions) repository.

### `If the cost is between X and Y.ipynb` 
(**Distributed Alignment Tutorial**) This is a tutorial reproducing one of the main experiments in [the Boundless DAS paper](https://arxiv.org/abs/2305.08809). Different from the first tutorial, this one involves trainable interventions that actively search for alignments with model internals.

### `Hook with new model and intervention types.ipynb` 
(**Intervention Tutorial**) This is a tutorial on integrating new model types with this library as well as customized interventions. We try to add `flan_t5` as well as a simple additive intervention. This tutorial covers a simple experiment as in `The capital of Spain is.ipynb`.


## System Requirements
- Python 3.8 is supported.
- Pytorch Version: 1.13.1
- Transfermers Minimum Version: 4.28.0.dev0
- Datasets Version: Version: 2.3.2

### `bf16` training with customized pytorch files
To save memory and speed up the training, we allow `bf16` mode when finding alignments. Since we rely on `torch` orthogonalization library, it does not support `bf16`. So, we did some hacks in the `torch` file to enable this. Modified files are in the `torch3.8_overwrite/*.py` folder. Currently, you need to replace these two files by hand in your environment. Here are two example directories for these two files:
```
/lib/python3.8/site-packages/torch/nn/utils/parametrizations.py
/lib/python3.8/site-packages/torch/nn/init.py
```

## Related Works in Discovering Causal Mechanism of LLMs
If you would like to read more works on this area, here is a list of papers that try to align or discover the causal mechanisms of LLMs. 
- [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997): This paper introduces interchange intervention (a.k.a. activation patching or causal scrubbing). It tries to align a causal model with the model's representations.
- [Inducing Causal Structure for Interpretable Neural Networks](https://arxiv.org/abs/2112.00826): Interchange intervention training (IIT) induces causal structures into the model's representations.
- [Localizing Model Behavior with Path Patching](https://arxiv.org/abs/2304.05969): Path patching (or causal scrubbing) to uncover causal paths in neural model.
- [Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/abs/2304.14997): Scalable method to prune out a small set of connections in a neural network that can still complete a task.
- [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593): Path patching plus posthoc representation study to uncover a circuit that solves the indirect object identification (IOI) task.
- [Rigorously Assessing Natural Language Explanations of Neurons](https://arxiv.org/abs/2309.10312): Using causal abstraction to validate [neuron explanations released by OpenAI](https://openai.com/research/language-models-can-explain-neurons-in-language-models).


## Citation
If you use this repository, please consider to cite relevant papers:
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
[^ii]: [Geiger et al. (2021a)](https://arxiv.org/abs/2106.02997), [Geiger et al. (2021b)](https://arxiv.org/abs/2112.00826), [Geiger et al. (2023)](https://arxiv.org/abs/2301.04709), [Wu et al. (2023)](https://arxiv.org/pdf/2303.02536)
