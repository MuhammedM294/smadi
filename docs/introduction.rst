
=====
SMADI
=====

    Soil Moisture Anomaly Detection Indicators


SMADI is a workflow designed to compute climate normals and detect anomalies for satellite soil moisture data, with a primary focus on `ASCAT <https://hsaf.meteoam.it/Products/ProductsList?type=soil_moisture>`_ surface soil moisture (SSM) products. The climatology, or climate normals, is computed to establish the distribution of SSM for each period and location. Subsequently, anomalies are computed accordingly.

The core objective of SMADI is to leverage these anomaly indicators to identify and highlight extreme events such as droughts and floods, providing valuable insights for environmental monitoring and management. Additionally, SMADI is applicable to various meteorological variables.

The following indices are provided:

-        **Z-score**: The Standardized Z-score

          z_score = (x - μ) / σ

                    where:
                    x: the average value of the variable in the time series data. It can be any of the following:
                    Daily average, weekly average, monthly average, etc.
                    μ: the long-term mean of the variable(the climate normal).
                    σ: the long-term standard deviation of the variable.

        
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



Features
========


-        **Data Reading**:  Read and preprocess the input data from Supported data sources. :mod:`smadi.data_reader`
-        **Climatology**: Compute the climatology for the input data based different time steps (e.g., monthly, dekadly, weekly, etc.). :mod:`smadi.climatology`
-        **Anomaly Detection**: Detect anomalies based on the computed climatology using different anomaly detection indices. :mod:`smadi.anomaly_detectors`
-        **Visualization**: Visualize the computed climatology and anomalies as time series, maps, and histograms. :mod:`smadi.plot , smadi.map`


Notebooks
=========
The following documentation is created from ipython notebooks in ``smadi/docs/examples``.