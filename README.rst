.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

.. image:: https://readthedocs.org/projects/smadi/badge/?version=latest
   :alt: ReadTheDocs
   :target: https://smadi.readthedocs.io/en/latest/README.html

.. image:: https://img.shields.io/pypi/v/smadi.svg
   :alt: PyPI-Server
   :target: https://pypi.org/project/smadi/

.. image:: https://mybinder.org/badge_logo.svg
   :alt: Binder
   :target: https://mybinder.org/v2/gh/MuhammedM294/SMADI_Tutorial/main?labpath=Tutorial.ipynb

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
   :alt: Project generated with PyScaffold
   :target: https://pyscaffold.org/

=====
SMADI
=====

    Soil Moisture Anomaly Detection Indicators

This repository contributes to a visiting research activity within the framework of `EUMETSAT HSAF <https://hsaf.meteoam.it/>`_, hosted by `TU Wien <https://www.tuwien.at/mg/geo>`_, on the subject "development of workflows for climate normal and anomaly calculation for satellite soil moisture products".

SMADI is a comprehensive workflow designed to compute climate normals and detect anomalies in satellite soil moisture data. The primary focus is on `ASCAT <https://hsaf.meteoam.it/Products/ProductsList?type=soil_moisture>`_ surface soil moisture (SSM) products. By establishing the distribution of SSM for each period and location, SMADI computes climatology, or climate normals, and subsequently identifies anomalies.

The core objective of SMADI is to leverage these anomaly indicators to identify and highlight extreme events such as droughts and floods, providing valuable insights for environmental monitoring and management. Furthermore, the methods used apply to other meteorological variables, such as precipitation, temperature, and more.


Features
========


-         **Data Reading**:  Read and preprocess the input data from Supported data sources. :mod:`smadi.data_reader` 

-        **Climatology**: Compute the climatology for the input data based on different time steps (e.g., monthly, dekadal, weekly, etc.). :mod:`smadi.climatology`

-        **Anomaly Detection**: Detect anomalies based on the computed climatology using different anomaly detection indices. :mod:`smadi.anomaly_detectors`

-        **Visualization**: Visualize the computed climatology and anomalies as time series, maps, and histograms. :mod:`smadi.plot , smadi.map`



Workflow Processing
-------------------

The package installation through pip will enable a command-line entry point for calculating anomalies using one or more of the available methods across various dates. The command, named 'smadi_run', is designed to compute indices for the ASCAT gridded NetCDF datasets. This Python entry point is intended to be executed through a bash shell command:

.. code-block::

   smadi_run <positional arguments> <options>

For more information about the positional and optional arguments of this command, run:

.. code-block::

   smadi_run -h 

Installation
------------

User Installation
~~~~~~~~~~~~~~~~~

For users who simply want to use `smadi`, you can install it via pip:

.. code-block:: 

    pip install smadi


Developer Installation
~~~~~~~~~~~~~~~~~~~~~~

If you're a developer or contributor, follow these steps to set up `smadi`:

1. Clone the repository:

.. code-block:: 

    git clone https://github.com/MuhammedM294/smadi

2. Navigate to the cloned directory:

.. code-block:: 

    cd smadi

3. Create and activate a virtual environment using Conda or virtualenv:

For Conda:

.. code-block:: 

    conda create --name smadi_env python=3.8
    conda activate smadi_env

For virtualenv:

.. code-block:: 

    virtualenv smadi_env
    source smadi_env/bin/activate  # On Unix or MacOS
    .\smadi_env\Scripts\activate    # On Windows

4. Install dependencies from requirements.txt:

.. code-block::

    pip install -r requirements.txt


Docker Usage
------------

To use the Docker image for SMADI, follow these steps:

1. **Pull the Docker Image:**

   Open your terminal and run the following command to pull the Docker image from Docker Hub:

   .. code-block:: bash

      docker pull muhammedabdelaal/smadi:latest

2. **Run the Docker Image:**

   After pulling the image, you can run it with the following command:

   .. code-block:: bash

      docker run -it muhammedabdelaal/smadi:latest

   This will start a container with the SMADI application.



.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
