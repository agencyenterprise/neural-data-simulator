# Neural Data Simulator

[![Linting](https://github.com/agencyenterprise/neural-data-simulator/actions/workflows/lint.yml/badge.svg)](https://github.com/agencyenterprise/neural-data-simulator/actions/workflows/lint.yml)
[![Tests](https://github.com/agencyenterprise/neural-data-simulator/actions/workflows/test.yml/badge.svg)](https://github.com/agencyenterprise/neural-data-simulator/actions/workflows/test.yml)

The Neural Data Simulator is a real-time system for generating electrophysiology data from behavioral data (e.g. cursor movement, arm kinematics, etc) in real-time. The NDS system can be used to test and validate closed-loop brain-computer interface systems without the need for a human in the loop, generate synthetic data for algorithm optimization, and provide a platform on which to develop BCI decoders.

## Documentation

See the [documentation](https://agencyenterprise.github.io/neural-data-simulator/) for a complete system overview, installation instructions, and API details.

## Installation

Ensure that Python `>=3.9` and `<3.12` is installed. Then, proceed to install LSL:

```
# on Linux/WSL
conda install -c conda-forge liblsl

# on macOS
brew install labstreaminglayer/tap/lsl

# on Windows
# should be installed automatically by pip when installing NDS
```

Install the NDS package with the included examples and utilities via pip:

```
pip install "neural-data-simulator[extras]"
```

## Quick start

Run the following scripts:

```
nds_post_install_config
run_closed_loop
```
![quick-start](https://raw.githubusercontent.com/agencyenterprise/neural-data-simulator/main/docs/source/images/quick-start.gif)

> **_NOTE:_** You might need to give permissions like network access when running the scripts.
