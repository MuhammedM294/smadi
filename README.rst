.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

.. image:: https://readthedocs.org/projects/smadi/badge/?version=latest
    :alt: ReadTheDocs
    :target: https://smadi.readthedocs.io/en/stable/

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

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
