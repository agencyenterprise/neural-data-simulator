# Installation

`NDS` requires Python `>=3.9` and `<3.12` to be installed. You can find which Python version you have installed by running `python --version` in a terminal. If you don't have Python installed, or you are running an unsupported version, you can download it from [python.org](https://www.python.org/downloads/). Python environment managers like pyenv, conda, and poetry are all perfectly suitable as well.

The `neural-data-simulator` package contains:
- core components: [encoder](encoder.md) and [electrophysiology generator](ephys_generator.md)
- example implementations: [neural decoder](decoders.md) and [center-out reaching task](tasks.md)
- [utilities](utilities.md): [LSL](#lab-streaming-layer) streamer and recorder

You can install `NDS`, with all included utilities and example implementations, using:

```
pip install neural-data-simulator[extras]
```

```{important}
The `pip` installer will copy the NDS scripts to a predefined folder for binaries. For example, on Linux this can be `$HOME/.local/bin`. Make sure that the this folder is added to the `PATH` variable in order to be able to execute the NDS scripts without having to enter the full path. Usually the installer will print a warning if the bin folder is not added to the PATH.
```

If you prefer to install `NDS` with the strictly required dependencies for its core functionality then execute the following command:

```
pip install neural-data-simulator
```

```{note}
The `extras` installation is required to run the [BCI closed loop example](running_bci.md).
```

## Lab Streaming Layer

`NDS` requires [LSL](https://labstreaminglayer.readthedocs.io/index.html) for data input, output, and internal data transfer.
[LSL](https://labstreaminglayer.readthedocs.io/index.html) can be installed by running:

```
# on Linux/WSL
conda install -c conda-forge liblsl

# on macOS
brew install labstreaminglayer/tap/lsl

# on Windows
# the python package should be installed automatically by pip when installing the dependencies.
```

On Windows, the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) has to be installed.

```{note}
If the command above is not able to find a liblsl package for your system, you can build it manually [following the guide or running the script](https://github.com/sccn/liblsl#building-liblsl).
```

## Matplotlib

If NDS was installed with the `[extras]` dependencies, [matplotlib](https://matplotlib.org) will also be included. This library is being used to display velocities and trajectories plots at the end of running the BCI closed loop simulation. In order to display figures, `matplotlib` can use different rendering backends. On some Linux distributions and on macOS, `matplotlib` might work out of the box, but other Linux distributions require explicitly installing a backend like `PyQt`. On Windows, it also requires the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) to be installed. Please consult the [matplotlib documentation](https://matplotlib.org/stable/devel/dependencies.html#optional-dependencies) for more information.

## Post-installation script

The purpose of this script is to prepare the environment with the default configuration files and sample data. You should manually execute this script once after installing the NDS package:

```

nds_post_install_config

```

By default `nds_post_install_config` will copy all the configuration files and download all sample data to the `$HOME/.nds` folder, skipping files that already exist. If you want to overwrite existing files, you can use the `--overwrite-existing-files` argument.

If you installed NDS without the `[extras]` dependencies, you can use the `--ignore-extras-config` argument to only copy the core components' default configuration.

```{note}
An internet connection is required to download the sample data. If you are working offline, you can skip the download by using the `--ignore-sample-data-download` argument.
```

## Visualization tools

For visualizing the data streams in real-time we recommend installing the open-ephys package with the LSL plugin and configuring it as described in [Visualizing Data](visualization.md).
