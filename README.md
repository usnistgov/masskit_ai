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
  * first install [Masskit](https://github.com/usnistgov/masskit).
  * in the masskit directory, run `source environments/init_masskit.sh -m`.
    * if you wish to use a cpu-only version of the library, run `source environments/init_masskit.sh -c` instead.
  * change to a directory that will hold the masskit_ai repository
  * `git clone https://github.com/usnistgov/masskit_ai.git`
  * `cd masskit_ai`
  * `pip install -e .`
 