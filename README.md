<br />
<div align="center">
  <h1 align="center"><img src="https://i.ibb.co/BNkhQH3/pyvene-logo.png"></h1>
  <a href="https://arxiv.org/abs/2403.07809"><strong>Read our paper »</strong></a> | <a href="https://stanfordnlp.github.io/pyvene/"><strong>Read the docs »</strong></a>
</div>     

<br />
<a href="https://pypi.org/project/pyvene/"><img src="https://img.shields.io/pepy/dt/pyvene?color=green"></img></a>
<a href="https://pypi.org/project/pyvene/"><img src="https://img.shields.io/pypi/v/pyvene?color=red"></img></a> 
<a href="https://pypi.org/project/pyvene/"><img src="https://img.shields.io/pypi/l/pyvene?color=blue"></img></a>

# A Library for _Understanding_ and _Improving_ PyTorch Models via Interventions

**pyvene** is an open-source Python library for intervening on the internal states of
PyTorch models. Interventions are an important operation in many areas of AI, including
model editing, steering, robustness, and interpretability.

pyvene has many features that make interventions easy:

- Interventions are the basic primitive, specified as dicts and thus able to be saved locally
  and shared as serialisable objects through HuggingFace.
- Interventions can be composed and customised: you can run them on multiple locations, on arbitrary
  sets of neurons (or other levels of granularity), in parallel or in sequence, on decoding steps of
  generative language models, etc.
- Interventions work out-of-the-box on any PyTorch model! No need to define new model classes from
  scratch and easy interventions are possible all kinds of architectures (RNNs, ResNets, CNNs, Mamba).

pyvene is under active development and constantly being improved 🫡


> [!IMPORTANT]
> Read the pyvene docs at [https://stanfordnlp.github.io/pyvene/](https://stanfordnlp.github.io/pyvene/)!


## Installation

To install the latest stable version of pyvene:

```
pip install pyvene
```

Alternatively, to install a bleeding-edge version, you can clone the repo and install:

```
git clone git@github.com:stanfordnlp/pyvene.git
cd pyvene
pip install -e .
```

When you want to update, you can just run `git pull` in the cloned directory.

We suggest importing the library as:

```
import pyvene as pv
```

## Citation
If you use this repository, please consider to cite our library paper:
```bibtex
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
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=stanfordnlp/pyvene,stanfordnlp/pyreft&type=Date)](https://star-history.com/#stanfordnlp/pyvene&stanfordnlp/pyreft&Date)

