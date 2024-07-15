# nonnadtivity package source code

This is the core code for the nonadditivity analysis package.
Below, see a list of which file handles which part of the functionality.

The whole analysis uses [Pandas Dataframes](https://pandas.pydata.org/).

## what is where

- [`cli.py`](cli.py): Has the main function that is called by using the command line command `nonadditivity`. calls a function defined in the [nonadditivity_workflow.py](nonadditivity_workflow.py).
- [`naa_workflow`](naa_workflow): Contains the core Nonadditivity algorithm as well as some I/O functionality.
- [`classification`](classification): contains all functionality to run the classification of Nonadditivity cases.
- [`utils`](utils): contains helper scripts.
