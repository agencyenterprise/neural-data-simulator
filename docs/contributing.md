---
Note: "Are you viewing this document on GitHub? For the best experience, view it on the website https://agencyenterprise.github.io/neural-data-simulator/contributing.html."
---

# Contributing

We warmly welcome you to contribute your feedback, code improvements, suggestions, and questions on the Neural Data Simulator.
Please contribute to NDS by creating GitHub issues or by submitting pull requests.

## Reporting issues

Feel free to open an issue if you would like to discuss a new feature request or report a bug. When creating a bug report, please include as much information as possible to help us reproduce the bug as well as what the actual and expected behavior is.

## Contributing code

If you've already setup your development environment and are able to run the make targets for linting and running unit tests, you can skip to the [code requirements and conventions](#code-requirements-and-conventions) section.

### Preparing your environment

Start by cloning the repository:

```
git clone https://github.com/agencyenterprise/neural-data-simulator.git
cd neural-data-simulator
```

This project requires Python `>=3.9` and `<3.12` to be installed. You can find the Python version you have installed by running `python --version` in a terminal. If you don't have Python installed or are running an unsupported version, you can download a supported version from [python.org](https://www.python.org/downloads/).

We use [poetry](https://python-poetry.org/) to manage dependencies and virtual environments. Follow the instructions from [poetry's documentation](https://python-poetry.org/docs/#installation) to install it if you don't have it on your system.

Install the dependencies by running the following command in a shell within the project directory:

```
poetry install
```

This will resolve and install the dependencies from `poetry.lock` and will install the `neural-data-simulator` package in editable mode.

```{note}
On Windows you might have to install [Microsoft C++ Build Tools - Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to run the command above successfully.
```

To bootstrap configurations, models and sample data, run:

```
poetry run nds_post_install_config
```

Additionally, you will need to install [LSL](https://labstreaminglayer.readthedocs.io/index.html). This can be done by running:

```
# on Linux/WSL
conda install -c conda-forge liblsl

# on macOS
brew install labstreaminglayer/tap/lsl

# on Windows
# the python package should be installed automatically by pip when installing the dependencies.
```

On Windows, the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) has to be installed.

For building the documentation or running spellchecking you will need to install the [enchant C library](https://abiword.github.io/enchant/) if it's not already installed on your system. To do this please follow the instructions on the [pyenchant documentation page](https://pyenchant.github.io/pyenchant/install.html#installing-the-enchant-c-library).

### Using the environment

If you are not already using a virtual environment, `poetry` will [create one for you by default](https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment). You will need to use this virtual env when using or working on the package.

You can do that in one of two ways:

1.  Activate the environment directly via:

    ```
    poetry shell
    ```

    ```{note}
       On Windows, if not using the Windows Subsystem for Linux (WSL), you will need to enable `Powershell` to execute scripts via [Set-ExecutionPolicy](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.security/set-executionpolicy?view=powershell-7.1).

       Before executing `poetry shell`, open a `Powershell` window as administrator and run:

         Set-ExecutionPolicy RemoteSigned

    ```

    Now you are able to execute scripts that use the python environment, like:

    ```
    encoder
    ```

2.  Alternatively, you can prepend `poetry run` to any python or poetry command in order to run the command within the virtual environment instead of needing to activate the environment. Example:
    ```
    poetry run encoder
    ```

```{note}
By default, the encoder expects to consume an LSL stream, if you prefer to read from a file set this [configuration](configuring.md#prerecorded-behavior-file).
```

If you are already using your own virtual environment, you should not need to change anything.

## Process

### Versioning

NDS uses [semantic versioning](https://en.wikipedia.org/wiki/Software_versioning#Semantic_versioning) to identify its releases.

We use the [release on push](https://github.com/rymndhng/release-on-push-action/tree/master/) github action to generate the new version for each release. This github action generates the version based on a pull request label assigned before merge. The supported labels are:

- `release-patch`
- `release-minor`
- `release-major`
- `norelease`

### Automatic release

Merged pull requests with one of the labels `release-patch`, `release-minor` or `release-major` will trigger a release job on CI.

The release job will:

1. generate a new package version using semantic versioning provided by [release on push](https://github.com/rymndhng/release-on-push-action/tree/master/)
1. update the `pyproject.toml` version using `poetry`
1. commit the updated `pyproject.toml` file using the [git-auto-commit action](https://github.com/stefanzweifel/git-auto-commit-action/tree/v4/)
1. push the package to pypi using [poetry publish](JRubics/poetry-publish@v1.16)
1. build a new docker image and tag it with the previously generated semantic version

Pull requests merged with the tag `norelease` will not trigger any of the actions listed above.

## Code requirements and conventions

```{note}
The following commands require `GNU make` to be installed, on Windows you can install it with [Chocolatey](https://chocolatey.org/install):
   `choco install make`
```

Before opening a pull request, please make sure that all of the following requirements are met:

1. all unit and integration tests are passing:
   ```
   make test
   ```
2. the code is linted and formatted:
   ```
   make lint
   ```
3. spelling is checked:
   ```
   make spellcheck
   ```
4. the documentation builds without warnings:
   ```
   make htmldoc
   ```
5. type hinting is used on all function and method parameters and return values, excluding tests
6. docstring usage conforms to the following:
   1. all docstrings should follow [PEP257 Docstring Conventions](https://peps.python.org/pep-0257/)
   2. all public API classes, functions, methods, and properties have docstrings and follow the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
   3. docstrings on private objects are not required, but are encouraged where they would significantly aid understanding
7. testing is done using the pytest library, and test coverage should not unnecessarily decrease.
