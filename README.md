# Course Quantum Machine Learning

This repository contains labs in the form of Jupyter Notebooks.
To run the notebooks, you first need to set up an appropriate Python environment.

## Requirements

Please make sure the following software is installed:

-   Python ≥ 3.10
-   pip (usually installed together with Python)
-   Optional but recommended: Git (to download the latest course material)
-   Optional but recommended: VS Code or PyCharm Professional (for viewing, running, and editing the notebooks)

You can check your installation with:

``` bash
python --version
```

It may be necessary to specify the correct path to your Python installation.
If multiple Python versions are installed, Python might need to be called with

``` bash
python3 --version
```

or

``` bash
python3.10 --version
```

## Cloning the Repository

To clone the repository, copy the link shown under `Clone` and run:

``` bash
git clone <repository-url>
cd <repository-name>
```

Alternatively, the repository can be downloaded as a ZIP file from GitLab or Ilias and then extracted.

## Installation with venv (recommended)

The recommended method is to use a virtual environment with `venv`(standard since Python 3.3).

### 1. Create a Virtual Environment

#### macOS / Linux

``` bash
python3 -m venv .venv
```

#### Windows (PowerShell)

``` powershell
python -m venv .venv
```

This will create a folder called `.venv` in the project directory.

### 2. Activate the Environment

#### macOS / Linux

``` bash
source .venv/bin/activate
```

#### Windows (PowerShell)

``` powershell
.\.venv\Scripts\Activate.ps1
```

If you see an error message related to the execution policy:

``` powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Requirements

After activating the environment:

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

During the course, the requirements may change. If that happens, simply run

``` bash
pip install -r requirements.txt
```

again after the `requirements.txt` file has been updated.

## Alternative: Installation with Conda

If you are using Anaconda or Miniconda:

### Create Environment

``` bash
conda create -n course python=3.11
```

### Activate

``` bash
conda activate course
```

### Install Requirements

``` bash
pip install -r requirements.txt
```

## Running the Notebooks

There are several ways to open and run the notebooks.

### 1. Using Visual Studio Code (recommended)

1.  Start VS Code
2.  Select "Open Folder" → choose the project folder
3.  If necessary: install the Python extension
4.  Select the Python interpreter in the bottom right (`.venv`)

You can then open a notebook by clicking on it.
Cells can be executed using the play icon or with **Shift + Enter**.

If the environment is not detected automatically:

-   `Ctrl + Shift + P`
-   "Python: Select Interpreter"
-   Select `.venv`

### 2. Using PyCharm Professional

JetBrains IDEs support Jupyter Notebooks directly.

Steps:

1.  Open the project
2.  Under *Settings → Python Interpreter*, select `.venv`
3.  Open the notebook file (`.ipynb`)
4.  Run cells using the play button or **Shift + Enter**

Note:
The Community Edition of PyCharm does not support Jupyter Notebooks
directly.
You need the licensed PyCharm Professional version, which you can obtain
using your student account.

### 3. Using JupyterLab

After activating the environment:

``` bash
jupyter lab
```

A browser window will open automatically.
You can open the desired notebook (`.ipynb`) there and execute cells with **Shift + Enter**.

If `jupyter lab` is not found, install it first:

``` bash
pip install jupyterlab
```

## Deactivating the Environment

Enter the following command in the terminal:

``` bash
deactivate
```
