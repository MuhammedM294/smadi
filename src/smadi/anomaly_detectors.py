"""
Soil Moisture Anomalies Calculation Module

This module provides a suite of methods for calculating soil moisture anomalies based on climatological data. 
The implemented methods include:

1. Z-Score: Standardized z-score method.
2. SMAPI (Soil Moisture Anomaly Percent Index): Measures anomalies as a percentage deviation from the climatological mean or median.
3. SMDI (Soil Moisture Deficit Index): Quantifies soil moisture deficit based on deviations from climatological median and extremes.
4. ESSMI (Empirical Standardized Soil Moisture Index): Uses a nonparametric empirical probability density function for standardizing soil moisture values.
5. SMAD (Standardized Anomaly Absolute Deviation): Calculates anomalies using the median and interquartile range, providing robustness to outliers.
6. SMDS (Soil Moisture Drought Severity): Assesses drought severity based on percentile rankings of soil moisture values.
7. SMCI (Soil Moisture Condition Index): Measures soil moisture content relative to climatological minima and maxima.
8. SMCA (Soil Moisture Content Anomaly): Quantifies anomalies as a deviation from climatological mean or median relative to maxima.
9. ParaDis (Parametric Distribution): Fits observed data to parametric distributions (e.g., beta, gamma).

"""

import warnings
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import matplotlib.pyplot as plt


from smadi.climatology import Climatology
from smadi.preprocess import filter_df, clim_groupping

from smadi.indicators import (
    zscore,
    smapi,
    smdi,
    smad,
    smca,
    smci,
    smds,
    essmi,
    smd,
    para_dis,
)


# Disable RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


class AnomalyDetector(ABC):
    """
    An abstract class for detecting anomalies in time series data based on the climatology.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        variable: str,
        fillna: bool = False,
        fillna_window_size: int = None,
        smoothing=False,
        smooth_window_size=None,
        timespan: List[str] = None,
        time_step: str = "month",
        normal_metrics: List[str] = ["mean"],
        agg_metric: str = "mean",
        dist: List[str] = None,
    ):
        """
        Initialize the AnomalyDetector class.

        parameters:
        -----------
        df: pd.DataFrame
            A dataframe containing the time series data.

        variable: str
            The name of the variable in the time series data to be analyzed.

        fillna: bool
            A boolean value to indicate whether to fill the missing values in the time series data.

        fillna_window_size: int
            The window size to be used for filling the missing values in the time series data.

        smoothing: bool
            A boolean value to indicate whether to smooth the time series data.

        smooth_window_size: int
            The window size to be used for smoothing the time series data.

        timespan: list[str, str] optional
            The start and end dates for a timespan to be aggregated. Format: ['YYYY-MM-DD ]

        normal_metrics: List[str]
            A list of metrics to be used in the climate normal(climatology) computation as reference values for the anomaly detection.
            It can be any of the following:
            ['mean', 'median', 'min', 'max']

        agg_metric: str
            A metric to be used for aggregating the time series data. It can be any of the following:
            ['mean', 'median', 'min', 'max', 'sum']

        dist: List[str]
            A list of parametric distribution to be used for fitting the observed data. It can be any of the following:
            ['beta', 'gamma']

        """
        self.df = df
        self.var = variable
        self.fillna = fillna
        self.fillna_window_size = fillna_window_size
        self.smoothing = smoothing
        self.smooth_window_size = smooth_window_size
        self.timespan = timespan
        self.time_step = time_step
        self.normal_metrics = normal_metrics
        self.agg_metric = agg_metric
        self.dist = dist
        self.anomalies_df = None

    @property
    def groupby_param(self):
        """
        The column name to be used for grouping the data for the anomaly detection.
        """

        return clim_groupping(self.clim_df, self.time_step)

    @property
    def clim_df(self):
        """
        The DataFrame containing the climate normal data.
        """
        clim_df = Climatology(
            self.df,
            self.var,
            self.fillna,
            self.fillna_window_size,
            self.smoothing,
            self.smooth_window_size,
            self.timespan,
            self.time_step,
            self.normal_metrics,
            self.agg_metric,
        ).compute_normals()
        return clim_df

    def apply_transformation(self, func, **kwargs) -> pd.DataFrame:
        self.anomalies_df = self.clim_df.copy()
        self.anomalies_df[func.__name__] = self.clim_df.groupby(self.groupby_param)[
            f"{self.var}-{self.agg_metric}"
        ].transform(func)
        return filter_df(self.anomalies_df, **kwargs)

    @abstractmethod
    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        pass


class ZScore(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Z-Score method.

    z_score = (x - μ) / σ

        where:
        x: the average value of the variable in the time series data. It can be any of the following:
        Daily average, weekly average, monthly average, etc.
        μ: the long-term mean of the variable(the climate normal).
        σ: the long-term standard deviation of the variable.

    """

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        return self.apply_transformation(zscore, **kwargs)


class SMAD(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Standardized Anomaly Absolute Deviation(SMAD) method.

    SMAD = (x - η) / IQR

    where:
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    η: the long-term median of the variable(the climate normal).
    IQR: the interquartile range of the variable. It is the difference between the 75th and 25th percentiles of the variable.

    """

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        self.normal_metrics = ["median"]
        return self.apply_transformation(smad, **kwargs)


class ESSMI(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Empirical Standardized Soil
    Moisture Index(ESSMI) method.

    The index is computed by fitting the nonparametric empirical probability
    density function (ePDF) using the kernel density estimator KDE

    f^h = 1/nh * Σ K((x - xi) / h)
    K = 1/√(2π) * exp(-x^2/2)

    where:
    f^h: the ePDF
    K: the Guassian kernel function
    h: the bandwidth of the kernel function as smoothing parameter (Scott's rule)
    n: the number of observations
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    xi: the ith observation

    The ESSMI is then computed by transforming the ePDF to the standard normal distribution with a mean of zero and
    a standard deviation of one using the inverse of the standard normal distribution function.

    ESSMI = Φ^-1(F^h(x))

        where:
        Φ^-1: the inverse of the standard normal distribution function
        F^h: the ePDF


    """

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        self.normal_metrics = ["mean"]
        return self.apply_transformation(essmi, **kwargs)


class SMDS(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Soil Moisture Drought Severity(SMDS) method.

    SMDS = 1 - SMP
    SMP = (rank(x) / (n+1))

    where:

    SMP: the Soil Moisture Percentile. It is the percentile of the average value of the variable in the time series data.
    SMDS: the Soil Moisture Drought Severity. It is the severity of the drought based on the percentile of the average value of the variable in the time series data.
    rank(x): the rank of the average value of the variable in the time series data.
    n: the number of years in the time series data.

    """

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        return self.apply_transformation(smds, **kwargs)


class SMCI(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Soil Moisture Condition Index(SMCI) method.

    SMCI = ((x - min) / (max - min))

    where:
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    min: the long-term minimum of the variable.
    max: the long-term maximum of the variable.

    """

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        self.normal_metrics = ["min", "max"]
        return self.apply_transformation(smci, **kwargs)


class SMAPI(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Soil Moisture Anomaly Percent Index(SMAPI) method.

    SMAPI = ((x - ref) / ref) * 100

    where:
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    ref: the long-term mean (μ​) or median (η) of the variable(the climate normal).

    """

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        self.anomalies_df = self.clim_df.copy()
        for metric in self.normal_metrics:
            self.anomalies_df[f"smapi-{metric}"] = self.clim_df.groupby(
                self.groupby_param
            )[f"{self.var}-{self.agg_metric}"].transform(smapi, metric=metric)

            self.anomalies_df[f"smapi-{metric}"] = self.anomalies_df[
                f"smapi-{metric}"
            ].clip(lower=-100, upper=100)

        return filter_df(self.anomalies_df, **kwargs)


class SMDI(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Soil Moisture Deficit Index(SMDI) method.

    SMDI = 0.5 * SMDI(t-1) + (SD(t) / 50)

    where

    SD(t) = ((x - η) / (η - min)) * 100 if x <= η
    SD(t) = ((x - η) / (max - η)) * 100 if x > η

    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    η: the long-term median of the variable(the climate normal).
    min: the long-term minimum of the variable.
    max: the long-term maximum of the variable.
    t: the time step of the time series data.

    """

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        self.anomalies_df = self.clim_df.copy()
        self.anomalies_df["sd"] = self.clim_df.groupby(self.groupby_param)[
            f"{self.var}-{self.agg_metric}"
        ].transform(smd)
        self.anomalies_df["smdi"] = smdi(self.anomalies_df["sd"])

        return filter_df(self.anomalies_df, **kwargs)


class SMCA(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Soil Moisture Content Anomaly(SMCA) method.

    SMCA = (x - ref) / (max - ref)

    where:
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.

    ref: the long-term mean (μ) or median (η) of the variable(the climate normal).
    max: the long-term maximum of the variable.
    min: the long-term minimum of the variable.

    """

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        self.anomalies_df = self.clim_df.copy()
        for metric in self.normal_metrics:
            self.anomalies_df[f"smca-{metric}"] = self.clim_df.groupby(
                self.groupby_param
            )[f"{self.var}-{self.agg_metric}"].transform(smca, metric=metric)

        return filter_df(self.anomalies_df, **kwargs)


class ParaDis(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on fitting the observed data to a parametric distribution(e.g. beta and gamma).

    """

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        self.anomalies_df = self.clim_df.copy()
        for dist in self.dist:
            self.anomalies_df[f"{dist}"] = self.clim_df.groupby(self.groupby_param)[
                f"{self.var}-{self.agg_metric}"
            ].transform(para_dis, dist=dist)

            self.anomalies_df[f"{dist}"] = self.anomalies_df[f"{dist}"].clip(
                lower=-3, upper=3
            )

        return filter_df(self.anomalies_df, **kwargs)


class AnomalyDetectorFactory:
    """
    A factory class for creating anomaly detectors based on the provided method.
    """

    methods = {
        "zscore": ZScore,
        "smapi": SMAPI,
        "smdi": SMDI,
        "smad": SMAD,
        "smca": SMCA,
        "smci": SMCI,
        "smds": SMDS,
        "essmi": ESSMI,
        "paradis": ParaDis,
    }

    @staticmethod
    def create_detector(method: str, **kwargs) -> AnomalyDetector:
        if method not in AnomalyDetectorFactory.methods:
            raise ValueError(f"Unknown method: {method}")
        return AnomalyDetectorFactory.methods[method](**kwargs)


# Supported anomaly detection methods
_Detectors = {
    "zscore": ZScore,
    "smapi-mean": SMAPI,
    "smapi-median": SMAPI,
    "smdi": SMDI,
    "smca-mean": SMCA,
    "smca-median": SMCA,
    "smad": SMAD,
    "smci": SMCI,
    "smds": SMDS,
    "essmi": ESSMI,
    "beta": ParaDis,
    "gamma": ParaDis,
}
