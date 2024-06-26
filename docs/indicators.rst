Anomaly Indicators
==================

The climatology is computed using a moving average window and averaging through multiple years with various time scales: monthly, bimonthly, decadal, and weekly. Anomalies are then detected using one or more of the following indices:

Z-score: The Standardized Z-score
---------------------------------


The Z-score is calculated using the formula:

.. math::

 z\_score = \frac{(x - \mu)}{\sigma}

where:

- **x**: The average (aggregated) value of the variable in the time series data based on the specified time step.
- **μ (mu)**: The long-term mean of the variable (the climate normal).
- **σ (sigma)**: The long-term standard deviation of the variable.


SMAD: Standardized Median Anomaly Deviation
-------------------------------------------
The SMAD is calculated using the formula:

.. math::

 SMAD = \frac{(x - \eta)}{IQR}

where:

- **x**: The average (aggregated) value of the variable in the time series data based on the specified time step.
- **η (eta)**: The long-term median of the variable (the climate normal).
- **IQR**: The interquartile range of the variable. It is the difference between the 75th and 25th percentiles of the variable.


Beta Distribution
-----------------

The beta distribution is implemented using the `scipy.stats` module in Python. The probability density function (PDF) of a Beta distribution is given by:

.. math::

 f(x; \alpha, \beta) = \frac{x^{\alpha-1} (1-x)^{\beta-1}}{B(\alpha, \beta)}

where:

- \( B(\alpha, \beta) \) is the Beta function, which normalizes the distribution.

In `scipy.stats`, the Beta distribution is implemented as `scipy.stats.beta(a, b)` where `a` and `b` correspond to the shape parameters α and β, respectively.


Gamma Distribution
---------------------

The gamma distribution is implemented using the `scipy.stats` module in Python. The PDF of a Gamma distribution is given by:

.. math::

 f(x; k, \theta) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \Gamma(k)}

where \( \Gamma(k) \) is the Gamma function.

In `scipy.stats`, the Gamma distribution is implemented as `scipy.stats.gamma(a, scale=θ)` where `a` corresponds to the shape parameter k, and `scale` corresponds to θ.

ESSMI: Empirical Standardized Soil Moisture Index
-------------------------------------------------

The index is computed by fitting the nonparametric empirical probability density function (ePDF) using the kernel density estimator (KDE):

.. math::

 \hat{f}_h = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)

where the kernel function \( K \) is given by:

.. math::

 K(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right)

and:

- \(\hat{f}_h\): the ePDF
- \( K \): the Gaussian kernel function
- \( h \): the bandwidth of the kernel function as a smoothing parameter (Scott's rule)
- \( n \): the number of observations
- \( x \): The average (aggregated) value of the variable in the time series data based on the specified time step.
- \( x_i \): the \( i \)-th observation

The ESSMI is then computed by transforming the ePDF to the standard normal distribution with a mean of zero and a standard deviation of one using the inverse of the standard normal distribution function:

.. math::

 ESSMI = \Phi^{-1}(\hat{F}_h(x))

where:

- \(\Phi^{-1}\): the inverse of the standard normal distribution function
- \(\hat{F}_h\): the ePDF

The kernel density estimator and the inverse of the standard normal distribution function can be implemented using the `scipy.stats` module in Python. The KDE can be computed using `scipy.stats.gaussian_kde`, and the inverse standard normal distribution can be obtained using `scipy.stats.norm.ppf`.

SMDS: Soil Moisture Drought Severity
------------------------------------

The SMDS is calculated using the formula:

.. math::

 SMDS = 1 - SMP

where the Soil Moisture Percentile (SMP) is given by:

.. math::

 SMP = \frac{\text{rank}(x)}{n + 1}

where:

- **SMP**: Soil Moisture Percentile. It is the percentile of the average value of the variable in the time series data.
- **SMDS**: Soil Moisture Drought Severity. It represents the severity of the drought based on the percentile of the average value of the variable in the time series data.
- **rank(x)**: The rank of the average value of the variable in the time series data.
- **n**: The number of years in the time series data.
- **x**: The average (aggregated) value of the variable in the time series data based on the specified time step.


SMCI: Soil Moisture Condition Index
-----------------------------------

The SMCI is calculated using the formula:

.. math::

 SMCI = \frac{(x - \text{min})}{(\text{max} - \text{min})}

where:

- **x**: The average (aggregated) value of the variable in the time series data based on the specified time step.
- **min**: The long-term minimum of the variable.
- **max**: The long-term maximum of the variable.


SMCA: Soil Moisture Content Anomaly
-----------------------------------

The SMCA is calculated using the formula:

.. math::

 SMCA = \frac{(x - \text{ref})}{(\text{max} - \text{ref})}

where:

- **x**: The average (aggregated) value of the variable in the time series data based on the specified time step.
- **ref**: The long-term mean (\( \mu \)) or median (\( \eta \)) of the variable (the climate normal).
- **max**: The long-term maximum of the variable.



SMAPI: Soil Moisture Anomaly Percentage Index
---------------------------------------------

A method for detecting anomalies in time series data based on the Soil Moisture Anomaly Percent Index (SMAPI) method.

The SMAPI is calculated using the formula:

.. math::

 SMAPI = \left( \frac{(x - \text{ref})}{\text{ref}} \right) \times 100

where:

- **x**: The average (aggregated) value of the variable in the time series data based on the specified time step.
- **ref**: The long-term mean (\( \mu \)) or median (\( \eta \)) of the variable (the climate normal).

SMDI: Soil Moisture Deficit Index
---------------------------------

The SMDI is calculated recursively using the formula:

.. math::

 SMDI(t) = 0.5 \times SMDI(t-1) + \left( \frac{SD(t)}{50} \right)

where:

- \( SD(t) \) is the Soil Moisture Deficit at time \( t \), defined as follows:

 .. math::

   SD(t) =
   \begin{cases}
     \frac{(x - \eta)}{(\eta - \text{min})} \times 100 & \text{if } x \leq \eta \\
     \frac{(x - \eta)}{(\text{max} - \eta)} \times 100 & \text{if } x > \eta \\
   \end{cases}

- \( x \) The average (aggregated) value of the variable in the time series data based on the specified time step.
- \( \eta \) is the long-term median of the variable (the climate normal).
- \( \text{min} \) is the long-term minimum of the variable.
- \( \text{max} \) is the long-term maximum of the variable.
- \( t \) is the time step of the time series data.






