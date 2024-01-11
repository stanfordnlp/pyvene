<br />
<div align="center">
  <h1 align="center"><img src="https://i.ibb.co/BC94vq1/intervention-library-icon-1.png" width="50" height="30">pyvene</h1>
  <a href="https://nlp.stanford.edu/~wuzhengx/"><strong>Library Paper and Doc Are Forthcoming »</strong></a>
</div>

# **pyvene: Use _Interventions_ to Learn Model's _Causal Mechanisms_**
**pyvene** supports customizable interventions on different neural architectures (e.g., RNN or Transformers). It supports complex intervention schemas (e.g., parallel or serialized interventions) and a wide range of intervention modes (e.g., static or trained interventions) at scale to gain interpretability insights.

**Getting Started:** [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/frankaging/pyvene/blob/main/tutorials/basic_tutorials/Basic_Intervention.ipynb) [**pyvene Intervention 101**]  

## Installation
Install this package directly from the source code as,
```bash
!pip install git+https://github.com/frankaging/pyvene.git
```

## _Wrap_ , _Intervene_ and _Share_
You can intervene with supported models as,
```python
import pyvene
from pyvene import IntervenableRepresentationConfig, IntervenableConfig, IntervenableModel

# provided wrapper for huggingface gpt2 model
_, tokenizer, gpt2 = pyvene.create_gpt2()

# turn gpt2 into intervenable_gpt2
intervenable_gpt2 = IntervenableModel(
    intervenable_config = IntervenableConfig(
        intervenable_representations=[
            IntervenableRepresentationConfig(
                0,            # intervening layer 0
                "mlp_output", # intervening mlp output
                "pos",        # intervening based on positional indices of tokens
                1             # maximally intervening one token
            ),
        ],
    ), 
    gpt2
)

# intervene base with sources on the fourth token.
original_outputs, intervened_outputs = intervenable_gpt2(
    tokenizer("The capital of Spain is", return_tensors="pt"),
    [tokenizer("The capital of Italy is", return_tensors="pt")],
    {"sources->base": ([[[4]]], [[[4]]])}
)
original_outputs.last_hidden_state - intervened_outputs.last_hidden_state
```
which returns,

```
tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0008, -0.0078, -0.0066,  ...,  0.0007, -0.0018,  0.0060]]])
```
showing that we have causal effects only on the last token as expected. You can share your interventions through Huggingface with others with a single call,
```python
intervenable_gpt2.save(
    save_directory="./your_gpt2_mounting_point/",
    save_to_hf_hub=True,
    hf_repo_name="your_gpt2_mounting_point",
)
```
We see interventions are knobs that can mount on models. And people can share their knobs with others to share knowledge about how to steer models. You can try this at [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/frankaging/pyvene/blob/main/tutorials/basic_tutorials/Load_Save_and_Share_Interventions.ipynb) [**Intervention Sharing**]  


## Selected Tutorials

| **Level** |  **Tutorial** |  **Run in Colab** |  **Description** |
| --- | -------------  |  -------------  |  -------------  | 
| Beginner |  [**Getting Started**](tutorials/basic_tutorials/Basic_Intervention.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/frankaging/pyvene/blob/main/tutorials/basic_tutorials/Basic_Intervention.ipynb)  |  Introduces basic static intervention on factual recall examples |
| Beginner | [**Intervened Model Generation**](tutorials/advanced_tutorials/Intervened_Model_Generation.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/frankaging/pyvene/blob/main/tutorials/advanced_tutorials/Intervened_Model_Generation.ipynb) | Shows how to intervene a model during generation |
| Intermediate | [**Intervene Your Local Models**](tutorials/basic_tutorials/Add_New_Model_Type.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/frankaging/pyvene/blob/main/tutorials/basic_tutorials/Add_New_Model_Type.ipynb) | Illustrates how to run this library with your own models |
| Advanced | [**Trainable Interventions for Causal Abstraction**](tutorials/advanced_tutorials/DAS_Main_Introduction.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/frankaging/pyvene/blob/main/tutorials/advanced_tutorials/DAS_Main_Introduction.ipynb) | Illustrates how to train an intervention to discover causal mechanisms of a neural model |


## Causal Abstraction: From Interventions to Gain Interpretability Insights
Basic interventions are fun we cannot make any causal claim systematically. To gain actual interpretability insights, we want to measure the counterfactual behaviors of a model in a data-driven fashion. In other words, if the model responds systematically to your interventions, then you start to associate certain regions in the network with a high-level concept. We also call this alignment search process with model internals.

### Understanding Causal Mechanisms with Static Interventions
Here is a more concrete example,
```py
def add_three_numbers(a, b, c):
    var_x = a + b
    return var_x + c
```
The function solves a 3-digit sum problem. Let's say, we trained a neural network to solve this problem perfectly. "Can we find the representation of (a + b) in the neural network?". We can use this library to answer this question. Specifically, we can do the following,

- **Step 1:** Form Interpretability (Alignment) Hypothesis: We hypothesize that a set of neurons N aligns with (a + b).
- **Step 2:** Counterfactual Testings: If our hypothesis is correct, then swapping neurons N between examples would give us expected counterfactual behaviors. For instance, the values of N for (1+2)+3, when swapping with N for (2+3)+4, the output should be (2+3)+3 or (1+2)+4 depending on the direction of the swap.
- **Step 3:** Reject Sampling of Hypothesis: Running tests multiple times and aggregating statistics in terms of counterfactual behavior matching. Proposing a new hypothesis based on the results. 

To translate the above steps into API calls with the library, it will be a single call,
```py
intervenable.evaluate(
    train_dataloader=test_dataloader,
    compute_metrics=compute_metrics,
    inputs_collator=inputs_collator
)
```
where you provide testing data (basically interventional data and the counterfactual behavior you are looking for) along with your metrics functions. The library will try to evaluate the alignment with the intervention you specified in the config.

--- 

### Understanding Causal Mechanism with Trainable Interventions
The alignment searching process outlined above can be tedious when your neural network is large. For a single hypothesized alignment, you basically need to set up different intervention configs targeting different layers and positions to verify your hypothesis. Instead of doing this brute-force search process, you can turn it into an optimization problem which also has other benefits such as distributed alignments.

In its crux, we basically want to train an intervention to have our desired counterfactual behaviors in mind. And if we can indeed train such interventions, we claim that causally informative information should live in the intervening representations! Below, we show one type of trainable intervention `models.interventions.RotatedSpaceIntervention` as,
```py
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
```
Instead of activation swapping in the original representation space, we first **rotate** them, and then do the swap followed by un-rotating the intervened representation. Additionally, we try to use SGD to **learn a rotation** that lets us produce expected counterfactual behavior. If we can find such rotation, we claim there is an alignment. `If the cost is between X and Y.ipynb` tutorial covers this with an advanced version of distributed alignment search, [Boundless DAS](https://arxiv.org/abs/2305.08809). There are [recent works](https://www.lesswrong.com/posts/RFtkRXHebkwxygDe2/an-interpretability-illusion-for-activation-patching-of) outlining potential limitations of doing a distributed alignment search as well.

You can now also make a single API call to train your intervention,
```py
intervenable.train(
    train_dataloader=train_dataloader,
    compute_loss=compute_loss,
    compute_metrics=compute_metrics,
    inputs_collator=inputs_collator
)
```
where you need to pass in a trainable dataset, and your customized loss and metrics function. The trainable interventions can later be saved on to your disk. You can also use `intervenable.evaluate()` your interventions in terms of customized objectives.


## Unit-tests
When adding new methods or APIs, unit tests are now enforced. To run existing tests, you can kick off the python unittest command in the discovery mode as,
```bash
cd pyvene
python -m unittest discover -p '*TestCase.py'
```
When checking in new code, please also consider to add new tests in the same PR. Please include test results in the PR to make sure all the existing test cases are passing. Please see the `qa_runbook.ipynb` notebook about a set of conventions about how to add test cases. The code coverage for this repository is currently `low`, and we are adding more automated tests.


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
