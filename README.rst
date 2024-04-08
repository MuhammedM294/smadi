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


Usage
-----

Here are some basic examples to demonstrate how to use `smadi`:

For more detailed usage instructions and parameters, you can refer to the notebook tutorial available at `link<https://github.com/MuhammedM294/SMADI_Tutorial/blob/main/Tutorial.ipynb>`_

For Single Point 
~~~~~~~~~~~~~~~~~

1. Import the package:

.. code-block:: 

    from smadi.data_reader import read_grid_point

2. Load your ASCAT data:

.. code-block:: 

    data_path = "path/to/data"

    loc = (lon,lat) 

    data = read_grid_point(loc=loc, ascat_sm_path=data_path, read_bulk=False)

    ascat_ts = data.get("ascat_ts")

    # Filter the time series to get only the soil moisture data
    sm_ts = ascat_ts.get("sm")


3. Compute the climatology 

.. code-block::

   from smadi.climatology import Climatology

   # Create a climatology object
   cl = Climatology(df=ascat_ts, variable="sm")

   # Set the time step for computing the climatology
   cl.time_step = "month"  # Supported time steps are "month", "bimonth", "dekad","week", "day"
   

   cl_df = cl.compute_normals()

   # Set mutiple metrics for computing the climatology
   cl.normal_metrics = ["mean", "median", "min", "max"]  # Default is ['mean']

   cl_df = cl.compute_normals()

4. Compute Anomalies

.. code-block::

   from smadi.anomaly_detectors import (
    ZScore,
    SMAPI,
    SMDI,
    SMCA,
    SMAD,
    SMCI,
    SMDS,
    ESSMI,
    ParaDis,
    AbsoluteAnomaly)


   # Zscore Usage Example
   zscore = ZScore( df=ascat_ts,
    variable="sm",
    fillna=True,
    fillna_window_size=3,
    smoothing=True,
    smooth_window_size=31,
    time_step="month",
   )
   anomaly_df = zscore.detect_anomaly()
   
   
   
   # SMAPI Usage Example
      smapi = SMAPI(
       df=ascat_ts,
       variable="sm",
       fillna=True,
       fillna_window_size=3,
       smoothing=True,
       smooth_window_size=31,
       time_step="month",
       normal_metrics=["mean", "median"],
      )
   
      anomaly_df = smapi.detect_anomaly()

For Country Scale Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To apply the workflow on a country scale, utilize the 'run' command specifying either the country name or bounding box (bbox) to set the area of interest (AOI) using coordinates (minlon, maxlon, minlat, maxlat).
For more information on the available arguments, you can run `run -h` command.

.. code-block::

     # Run the workflow for a country
     run "path/to/data" \
         "country_name" \
         "time_step" \
          --year <> \
          --month <>  \
          --method <>  \
          --save_to <>

     # Run the workflow for bbox
     run "path/to/data"\
         "minlon, maxlon, minlat, maxlat"\
         "time_step" \
         --year <> \
         --month <>  \
         --method <>  \
         --save_to <>



.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
