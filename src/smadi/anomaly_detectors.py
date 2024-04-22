"""
A module for soil moisture anomalies calculation methods based on climatology. The module implements the following methods:

1. Z-Score: The standarized z-score method. 
2. SMAPI: The Soil Moisture Anomaly Percent Index method.
3. SMDI: The Soil Moisture Deficit Index method.
4. ESSMI: The Empirical Standardized Soil Moisture Index method.
5. SMAD: The Standardized Median Absolute Deviation method.
6. SMDS: The Soil Moisture Drought Severity method.
7. SMCI: The Soil Moisture Condition Index method.
8. SMCA: The Soil Moisture Content Anomaly method.
9. ParaDis: The Parametric Distribution method.


"""

import warnings
from typing import List
import pandas as pd
import matplotlib.pyplot as plt


from smadi.climatology import Climatology
from smadi.preprocess import filter_df, clim_groupping
from smadi.plot import (
    plot_figure,
    plot_colmns,
    draw_hbars,
    plot_categories_count,
    get_plot_options,
    clss_counter,
)
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


class AnomalyDetector(Climatology):
    """
    A base class for detecting anomalies in time series data based on climatology.
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

        metrics: List[str]
            A list of metrics to be used in the climate normal(climatology) computation. It can be any of the following:
            ['mean', 'median', 'min', 'max']

        groupby_param: List[str]
            The column name to be used for grouping the data for the computation of the climate normal.

        """
        super().__init__(
            df,
            variable,
            fillna,
            fillna_window_size,
            smoothing,
            smooth_window_size,
            timespan,
            time_step,
            normal_metrics,
            agg_metric,
        )
        # self.clim_df = pd.DataFrame()
        # self.groupby_param = None

    @property
    def groupby_param(self):

        return clim_groupping(self.clim_df, self.time_step)

    def _preprocess(self, **kwargs) -> pd.DataFrame:
        """
        Preprocess the data before computing the anomalies.
        """
        self._validate_input()
        self.clim_df = self.compute_normals(**kwargs)
        return self.clim_df

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        """
        Base method for detecting anomalies in time series data based on climatology.

        parameters:
        -----------
        kwargs: str
            Date/time parameters to be used for filtering the data before computing the anomalies. It can be any of the following:
            ['year', 'month', 'week', 'dekad' , start_date, end_date]

        returns:
        --------

        pd.DataFrame
            A dataframe containing the computed anomalies.

        """
        self.clim_df = self._preprocess().copy()

        return filter_df(self.clim_df, **kwargs)

    def plot_anomaly(
        self,
        df=None,
        x_axis=None,
        colmns=None,
        thresholds=None,
        plot_hbars=True,
        plot_categories=True,
        plot_style="ggplot",
        **kwargs,
    ):
        """
        Plot the computed anomalies.

        parameters:
        -----------

        df: pd.DataFrame
            The dataframe containing the data to plot. if None, the computed anomalies will be used.

        x_axis: list
            The x-axis values for the plot. if None, the index of the dataframe will be used.

        colmns: dict
            A dictionary containing the columns to plot. The key is the column name and the value is the plot options.

        thresholds: str
            The name of the anomaly method to use its corresponding thresholds values.

        plot_hbars: bool
            Whether to plot the horizontal bars on the plot according to the thresholds of the anomaly method used.

        plot_categories: bool
            Whether to plot the number of values in each category of the anomaly method that fall within the thresholds.

        plot_style: str
            The plot style to use for the plot.

        kwargs: dict
            Additional parameters to be used for customizing the plot. It can be any of the following:

            ['title', 'xlabel', 'ylabel', 'legend', 'figsize', 'grid']

        """
        plt.style.use(plot_style)
        # Set values for kwargs based on provided values
        plot_params = get_plot_options(**kwargs)

        df = self.detect_anomaly() if df is None else df
        x_axis = df.index if x_axis is None else x_axis
        plt.figure(figsize=plot_params["figsize"])
        plot_colmns(df, x_axis, colmns)

        if plot_hbars:
            draw_hbars(thresholds, x_axis)
        if plot_categories:
            results = clss_counter(df, colmns, thresholds)
            plot_categories_count(x_axis, results, thresholds)

        plot_figure(plot_params)

    def plot_fill_bet(
        self, df=None, x_axis=None, colmn=None, plot_style="ggplot", **kwargs
    ):
        """
        Plot the computed anomalies using the fill_between method.

        parameters:
        -----------

        df: pd.DataFrame
            The dataframe containing the data to plot. if None, the computed anomalies will be used.

        x_axis: list
            The x-axis values for the plot. if None, the index of the dataframe will be used.

        colmn: str
            The column name to plot.

        plot_style: str
            The plot style to use for the plot.

        kwargs: dict
            Additional parameters to be used for customizing the plot. It can be any of the following:

            ['title', 'xlabel', 'ylabel', 'legend', 'figsize', 'grid']

        """

        plt.style.use(plot_style)
        plot_params = get_plot_options(**kwargs)
        df = self.detect_anomaly() if df is None else df
        x_axis = df.index if x_axis is None else x_axis
        plt.figure(figsize=plot_params["figsize"])

        plt.fill_between(
            df.index,
            df[colmn],
            where=df[colmn].values >= 0,
            color="blue",
            label="Wet",
        )

        plt.fill_between(
            df.index,
            df[colmn],
            where=df[colmn].values < 0,
            color="red",
            label="Dry",
        )

        # Add horizontal line at y=0
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

        plt.legend(loc="upper right", fontsize=10)

        plot_figure(plot_params)


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
        agg_metric: str = "mean",
    ):
        super().__init__(
            df,
            variable,
            fillna,
            fillna_window_size,
            smoothing,
            smooth_window_size,
            timespan,
            time_step,
            agg_metric=agg_metric,
        )

    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        return self.clim_df

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)

        self.clim_df["zscore"] = self.clim_df.groupby(self.groupby_param)[
            f"{self.var}-{self.agg_metric}"
        ].transform(zscore)

        return filter_df(self.clim_df, **kwargs)


class SMAPI(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Soil Moisture Anomaly Percent Index(SMAPI) method.

    SMAPI = ((x - ref) / ref) * 100

    where:
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    ref: the long-term mean (μ​) or median (η) of the variable(the climate normal).

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
        agg_metric: str = "mean",
        normal_metrics: str = ["mean"],
    ):
        super().__init__(
            df,
            variable,
            fillna,
            fillna_window_size,
            smoothing,
            smooth_window_size,
            timespan,
            time_step,
            normal_metrics,
            agg_metric=agg_metric,
        )

    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        return self.clim_df

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)

        for metric in self.normal_metrics:
            self.clim_df[f"smapi-{metric}"] = self.clim_df.groupby(self.groupby_param)[
                f"{self.var}-{self.agg_metric}"
            ].transform(smapi, metric=metric)

            self.clim_df[f"smapi-{metric}"] = self.clim_df[f"smapi-{metric}"].clip(
                lower=-100, upper=100
            )

        return filter_df(self.clim_df, **kwargs)


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

    def __init__(
        self,
        df: pd.DataFrame,
        variable: str,
        fillna: bool = False,
        fillna_window_size: int = None,
        smoothing=False,
        smooth_window_size=None,
        timespan: List[str] = None,
        time_step: str = "week",
        agg_metric: str = "mean",
    ):
        super().__init__(
            df,
            variable,
            fillna,
            fillna_window_size,
            smoothing,
            smooth_window_size,
            timespan,
            time_step,
            agg_metric=agg_metric,
        )

    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        return self.clim_df

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)

        self.clim_df["sd"] = self.clim_df.groupby(self.groupby_param)[
            f"{self.var}-{self.agg_metric}"
        ].transform(smd)
        self.clim_df["smdi"] = smdi(self.clim_df["sd"])

        return filter_df(self.clim_df, **kwargs)


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
    ):

        super().__init__(
            df,
            variable,
            fillna,
            fillna_window_size,
            smoothing,
            smooth_window_size,
            timespan,
            time_step,
            normal_metrics,
            agg_metric=agg_metric,
        )

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)
        for metric in self.normal_metrics:
            self.clim_df[f"smca-{metric}"] = self.clim_df.groupby(self.groupby_param)[
                f"{self.var}-{self.agg_metric}"
            ].transform(smca, metric=metric)

        return filter_df(self.clim_df, **kwargs)


class SMAD(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Standardized Median Absolute Deviation(SMAD) method.

    SMAD = (x - η) / IQR

    where:
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    η: the long-term median of the variable(the climate normal).
    IQR: the interquartile range of the variable. It is the difference between the 75th and 25th percentiles of the variable.

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
        agg_metric: str = "mean",
    ):
        super().__init__(
            df,
            variable,
            fillna,
            fillna_window_size,
            smoothing,
            smooth_window_size,
            timespan,
            time_step,
            ["median"],
            agg_metric=agg_metric,
        )

    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        return self.clim_df

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)
        self.clim_df["smad"] = self.clim_df.groupby(self.groupby_param)[
            f"{self.var}-{self.agg_metric}"
        ].transform(smad)

        return filter_df(self.clim_df, **kwargs)


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
        agg_metric: str = "mean",
    ):

        super().__init__(
            df,
            variable,
            fillna,
            fillna_window_size,
            smoothing,
            smooth_window_size,
            timespan,
            time_step,
            ["min", "max"],
            agg_metric=agg_metric,
        )

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)

        self.clim_df["smci"] = self.clim_df.groupby(self.groupby_param)[
            f"{self.var}-{self.agg_metric}"
        ].transform(smci)

        return filter_df(self.clim_df, **kwargs)


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
        agg_metric: str = "mean",
    ):
        super().__init__(
            df,
            variable,
            fillna,
            fillna_window_size,
            smoothing,
            smooth_window_size,
            timespan,
            time_step,
            agg_metric=agg_metric,
        )

    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        return self.clim_df

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)
        self.clim_df["smds"] = self.clim_df.groupby(self.groupby_param)[
            f"{self.var}-{self.agg_metric}"
        ].transform(smds)
        return filter_df(self.clim_df, **kwargs)


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
        agg_metric: str = "mean",
    ):
        super().__init__(
            df,
            variable,
            fillna,
            fillna_window_size,
            smoothing,
            smooth_window_size,
            timespan,
            time_step,
            ["mean"],
            agg_metric=agg_metric,
        )

    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        return self.clim_df

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)
        self.clim_df["essmi"] = self.clim_df.groupby(self.groupby_param)[
            f"{self.var}-{self.agg_metric}"
        ].transform(essmi)

        return filter_df(self.clim_df, **kwargs)


class ParaDis(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on fitting the observed data to a parametric distribution(e.g. beta, gamma, etc.).

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
        agg_metric: str = "mean",
        dist: List[str] = ["beta"],
    ):

        super().__init__(
            df,
            variable,
            fillna,
            fillna_window_size,
            smoothing,
            smooth_window_size,
            timespan,
            time_step,
            agg_metric=agg_metric,
        )
        self.dist = dist

    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        return self.clim_df

    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)
        for dist in self.dist:
            self.clim_df[f"{dist}"] = self.clim_df.groupby(self.groupby_param)[
                f"{self.var}-{self.agg_metric}"
            ].transform(para_dis, dist=dist)

            self.clim_df[f"{dist}"] = self.clim_df[f"{dist}"].clip(lower=-3, upper=3)

        return filter_df(self.clim_df, **kwargs)


if __name__ == "__main__":

    # from pathlib import Path
    # from smadi.data_reader import extract_obs_ts

    # ascat_path = Path("/home/m294/VSA/Code/datasets")

    # po = 4854801
    # sm_ts = extract_obs_ts(po, ascat_path, obs_type="sm", read_bulk=False)

    pass
