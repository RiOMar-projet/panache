panache
=======

**panache** is a Python package for detecting river plumes in gridded
geophysical satellite data (primarily suspended particulate matter, SPM, in
NetCDF format).

As input it takes a list of NetCDF files, a bathymetry mask, and a JSON run
configuration, in order to create plume masks, illustrate per-timestep maps,
write summary statistics, and build a quick animated overview of the
processed run.

Installation
------------

**panache** requires Python 3.10 or newer.

.. code-block:: bash

   git clone https://github.com/RiOMar-projet/panache.git
   cd panache
   pip install -e .

Quick Start
-----------

Create a config file, then point the ``panache`` command at it:

.. code-block:: bash

   panache example_zone_config.json

The command prints the path to the generated ``Results.csv`` file when the
run completes.

Minimal required config fields are ``zone`` (or ``parameters``),
``input_path``, ``bathymetry_path``, ``output_dir``, ``overwrite``, and
``gif``. See ``example_zone_config.json`` and ``example_parameter_config.json``
in the repository root for the two config modes.

Outputs
-------

Each run writes its results into the configured ``output_dir``:

.. code-block:: text

   panache-output/
   ├── Results.csv
   ├── GIF.gif
   └── MAPS/
       ├── [base_file_name]_plume_mask.png
       ├── [base_file_name]_plume_mask.csv
       └── [base_file_name]_statistics.csv

.. toctree::
   :maxdepth: 1
   :hidden:

   api
