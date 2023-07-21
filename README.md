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

### Requirements

- Masskit is installed using the package managers [`conda`](https://conda.io/) and [`mamba`](https://mamba.readthedocs.io/).
If you do not have either installed we recommend installing [mambaforge](https://github.com/conda-forge/miniforge#mambaforge). Detailed installation instructions for macOS and Linux can be found [here](https://github.com/robotology/robotology-superbuild/blob/master/doc/install-mambaforge.md).

### Windows

- Download the latest Masskit Windows zip file from the
[Releases](https://github.com/usnistgov/masskit_ai/releases) page.
- Extract the contents of the file to a directory on your computer by navigating explorer to the
Downloads folder, right clicking on the zip file, and selecting `Extract all...`.
- Run `Miniforge Prompt` or `Anaconda Prompt` from the Start menu depending on whether you
installed `mambaforge` or a `conda` distribution.
- In the prompt window, use `cd masskit*` to change to the directory starting with `masskit` in
the directory you extracted the downloads to.
- Run `call init_masskit.bat /cpu` to create the `masskit_ai_cpu` package environment.
  - Alternatively, if you know you have an Nvidia GPU of sufficient power
  run `call init_masskit.bat /ml` to create the `masskit_ai` package environment.
- Run `pip install masskit*.whl`.

Whenever using the programs in Masskit, please make sure you initialize the appropriate package environment:

- If you installed `mambaforge` run `Miniforge Prompt` from the Start menu otherwise run `Anaconda Prompt` from the Start menu.
- If you installed the CPU only version package environment run `conda activate masskit_ai_cpu`.
- If you installed the GPU version package environment run `conda activate masskit_ai`

### Linux or macOS

- first install [Masskit](https://github.com/usnistgov/masskit) using [these instructions](https://github.com/usnistgov/masskit#installation).
- Change to a directory that will hold the masskit_ai directory.
- Run `git clone https://github.com/usnistgov/masskit_ai.git`.
- Run `cd masskit_ai`.
- Run `pip install .`.

Whenever using the programs in Masskit, please make sure you have initialized the appropriate package environment:

- If you installed the CPU only version package environment run `conda activate masskit_ai_cpu`.
- If you installed the GPU only version package environment run `conda activate masskit_ai`.
