.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

.. image:: https://readthedocs.org/projects/smadi/badge/?version=latest
    :alt: ReadTheDocs
    :target: https://smadi.readthedocs.io/en/latest/readme.html

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


SMADI is a workflow designed to compute climate normals and detect anomalies for satellite soil moisture data, with a primary focus on `ASCAT <https://hsaf.meteoam.it/Products/ProductsList?type=soil_moisture>`_ surface soil moisture (SSM) products. The climatology, or climate normals, is computed to establish the distribution of SSM for each period and location. Subsequently, anomalies are computed accordingly.

The core objective of SMADI is to leverage these anomaly indicators to identify and highlight extreme events such as droughts and floods, providing valuable insights for environmental monitoring and management. Additionally, SMADI is applicable to various meteorological variables.

The following indices are provided:

-        **Z-score**: The Standardized Z-score
-        **SMAPI**: Soil Moisture Anomaly Percentage Index
-        **SMDI**: Soil Moisture Deficit Index
-        **SMDS**: Soil Moisture Drought Severity
-        **ESSMI**: Empirical Standardized Soil Moisture Index
-        **SMCI**: Soil Moisture Condition Index
-        **SMAD**: Standardized Median Absolute Deviation
-        **SMCA**: Soil Moisture Content Anomaly
-        **Beta Distribution** 
-        **Gamma Distribution**

     `Germany SM Anomaly Maps July 2021`


.. image:: https://github.com/MuhammedM294/SMADI_Tutorial/assets/89984604/a8b7abb5-9636-4e82-8152-877397a61c3b>
      :alt: Germany SM Anomaly Maps July 2021
      :align: center


Workflow Processing
-------------------

The package installation through pip, will enable a command-line entry point for calculating anomalies using one or more of the available methods across various dates. The command, named 'run', is designed to compute indices for gridded NetCDF datasets. This Python entry point is intended to be executed through a bash shell command:

.. code-block::

   run <positional arguments> <options>

For more information about the positional and optional arguments of this command, run:

.. code-block::

   run -h 

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



.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
