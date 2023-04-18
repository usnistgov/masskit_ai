![Masskit logo](src/masskit_ai/docs/_static/img/masskit_ai_logo.png)

--------------------------------------------------------------------------------

Masskit AI is a Python package that uses the [Masskit](https://github.com/usnistgov/masskit) Python module and [PyTorch](https://pytorch.org/) to do machine learning on mass spectra, small molecules and peptides.  Masskit AI includes:

- AI models for predicting the HCD spectra of peptides.
- AI models for computing the retention index of small molecules.

<!-- toc -->
## Table of Contents

- [Installation](#installation)
- [Usage and API Documentation](https://pages.nist.gov/masskit_ai)
- [The Team](https://chemdata.nist.gov/)
- [License](LICENSE.md)

<!-- tocstop -->

## Installation

On a Linux or macOS computer that has [Anaconda](https://www.anaconda.com/) and [Git](https://git-scm.com/) installed

- first install [Masskit](https://github.com/usnistgov/masskit).
- change to a directory that will hold the masskit_ai directory
- run `git clone https://github.com/usnistgov/masskit_ai.git`
- run `cd masskit_ai`
- run `pip install -e .`

On a Windows computer that has [Anaconda](https://www.anaconda.com/) installed,
run the following from the Anaconda prompt:

- first install [Masskit](https://github.com/usnistgov/masskit).
- Download the masskit_ai windows wheel file from the
[Releases](https://github.com/usnistgov/masskit_ai/releases) page.
- run `pip install masskit_ai-1.0.1-py3-none-any.whl` in the directory to which you downloaded the file.
