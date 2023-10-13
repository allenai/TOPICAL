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
streamlit run src/topical/app.py
```

### Human evaluation

To reproduce our sampling of biomedical entities for human evaluation, run:

```python
from topical import nlm
import random

descriptors = {descr.ui: descr for descr in nlm.fetch_mesh() if int(descr.date_created.year) >= 2013 and descr.max_tree_depth() >= 7}