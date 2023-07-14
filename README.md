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

Masskit is currently only supported in a Python conda environment. If you do not have one installed we highly recommend installing [mambaforge](https://github.com/conda-forge/miniforge#mambaforge)

On a Linux or macOS computer that has Conda and [Git](https://git-scm.com/) installed

- first install [Masskit](https://github.com/usnistgov/masskit).
- change to a directory that will hold the masskit_ai directory
- run `git clone https://github.com/usnistgov/masskit_ai.git`
- run `cd masskit_ai`
- run `pip install .`

On a Windows computer that has Conda installed, download the latest masskit windows zip file from the
[Releases](https://github.com/usnistgov/masskit_ai/releases) page and extract the contents. Then run the
following from the Conda prompt:

- change to the directory that contains the extracted files.
- if you have a GPU of sufficient power, run `call init_masskit.bat /ml`, otherwise run `call init_masskit.bat /cpu`
- run `pip install masskit-1.1.0-py3-none-any.whl masskit_ai-1.1.0-py3-none-any.whl`
