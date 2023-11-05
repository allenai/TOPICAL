[![ci](https://github.com/allenai/TOPICAL/actions/workflows/ci.yml/badge.svg)](https://github.com/allenai/TOPICAL/actions/workflows/ci.yml)

# 🪄📄 TOPICAL: TOPIC pages AutomagicaLly

A tool for automatically generated topic pages for a broad range of biomedical entities and concepts.

Live demo coming soon! To run locally, following the instructions below.

## Installation

This repository requires Python 3.11 or later.

### Installing with pip

First, activate a virtual environment. Then, install with `pip` right from GitHub:

```bash
pip install "git+https://github.com/allenai/TOPICAL.git"
```

### Installing from source

If you plan on modifying the code, please clone the repo

```bash
git clone "git+https://github.com/allenai/TOPICAL.git"
cd TOPICAL
```

and install it from source using pip:

```bash
pip install -e .
```

Or, if you prefer, using [Poetry](https://python-poetry.org/):

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
# E.g. for Linux, macOS, Windows (WSL)
curl -sSL https://install.python-poetry.org | python3 -

# Install the package with poetry
poetry install
```

## Usage

Once installed, you can launch the demo with:

```bash
ENTREZ_EMAIL="<your email>" \
ENTREZ_API_KEY="<your Entrez API key>" \
OPENAI_API_KEY="<your OpenAI API key>" \
streamlit run src/topical/app.py
```

> __Note__: `ENTREZ_EMAIL` and `ENTREZ_API_KEY` are optional but highly recommended. You can get an Entrez API key [here](https://ncbiinsights.ncbi.nlm.nih.gov/new-api-keys-for-the-e-utilities/).

### Human evaluation

To reproduce our sampling of biomedical entities for human evaluation, run:

```python
import random

from topical import nlm

descriptors = {
    descr.ui: descr
    for descr in nlm.fetch_mesh()
    if int(descr.date_created.year) >= 2013 and descr.max_tree_depth() >= 7
}
```
