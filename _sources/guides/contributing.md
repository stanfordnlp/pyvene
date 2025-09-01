# Contributing Guidelines

*Pull requests, bug reports, and all other forms of contribution are welcomed and highly encouraged!* :octocat:

### The PR or Issue Title Format
Whenever you open an issue or a PR, please use this title format
```
[Priority Tag] Short Title
```
For Priority Tag, you can use `[P0]`-`[P2]`, `[P0]` is the highest priority, which means everyone should stop working and focus on this PR. For Minor issues, use `[Minor]`. For bugs, please use `[Bug Fix]` and see below.

---

### 📕 Pull Requests

#### Uninstall pyvene from python library
It becomes tricky if you have `pyvene` installed while debugging with this codebase, since imports can be easily messed up. Please run,
```bash
pip uninstall pyvene
```

#### Unit Test Run Is A Must before Creating PRs
When adding new methods or APIs, unit tests are now enforced. To run existing tests, you can kick off the python unittest command in the discovery mode as,
```bash
cd pyvene
python -m unittest discover -p '*TestCase.py'
```
For specific test case, yoou can run
```bash
cd pyvene
python -m unittest tests.integration_tests.ComplexInterventionWithGPT2TestCase
```
When checking in new code, please also consider to add new tests in the same PR. Please include test results in the PR to make sure all the existing test cases are passing. Please see the `qa_runbook.ipynb` notebook about a set of conventions about how to add test cases. The code coverage for this repository is currently `low`, and we are adding more automated tests.

#### Format
```
**Descriptions**:

[Describe your PR Here]


**Testing Done**:

[Provide logs, screen-shots, and files that contain tests you have done]

```

### 🪲 Bug Reports and Other Issues
Go to issues, and open with a title formatted as,
```
[Bug Fix] Short Title
```
For external requests (i.e., you are not in our core dev team), please use,
```
[External] Short Title
```

### 📄 Documentation
If making changes to documentation (in `docs/source`, deployed to GitHub Pages), please test your changes locally
(ideally in a fresh Python environment):

```
pip install -r requirements.txt
pip install -r docs/requirements.txt
cd docs
make html
python -m http.server
```

Then navigate to [localhost:8000/build/html](http://localhost:8000/build/html).

### 📥 Larger Feature Requests
Please email us!