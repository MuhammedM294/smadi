"""
A module for calculating climatology (climate normal) for different time steps (month, dekad, week) based on time series data.
"""

from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

from smadi.plot import plot_ts
from smadi.preprocess import (
    fillna,
    smooth,
    filter_df,
    monthly_agg,
    dekadal_agg,
    weekly_agg,
    bimonthly_agg,
    compute_clim,
)


class Preprocessor:
    """
    A class for preprocessing the time series data before aggregation.

    Attributes:
    -----------

    df: pd.DataFrame
        The input DataFrame containing the time series data to be aggregated.

    variable: str
        The variable/column in the DataFrame to be aggregated.

    fillna: bool
        Fill NaN values in the time series data using a moving window average.

    fillna_window_size: int
        The size of the moving window for filling NaN values. It is recommended to be an odd number.

    smoothing: bool
        Smooth the time series data using a moving window average.

    smooth_window_size: int
        The size of the moving window for smoothing(n-days). It is recommended to be an odd number.

    timespan: list[str, str] optional
        The start and end dates for a timespan to be aggregated. Format: ['YYYY-MM-DD ]

    Methods:
    --------

    preprocess:

        Preprocess the time series data by resampling, truncating, filling NaN values, and smoothing.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        variable: str,
        fillna: bool = False,
        fillna_window_size: int = None,
        smoothing: bool = False,
        smooth_window_size: int = None,
        timespan: List[str] = None,
    ):
        self.df = df
        self.variable = variable
        self.fillna = fillna
        self.fillna_window_size = fillna_window_size
        self.smoothing = smoothing
        self.smooth_window_size = smooth_window_size
        self.timespan = timespan

    def preprocess(self):
        resampled_df = self._resample(self.df)
        truncated_df = self._truncate(resampled_df)
        filled_df = self._fillna(truncated_df)
        smoothed_df = self._smooth(filled_df)
        smoothed_df.dropna(inplace=True)
        return smoothed_df

    def _resample(self, df):
        return pd.DataFrame(df[self.variable]).resample("D").mean()

    def _truncate(self, df):
        if self.timespan:
            return df.truncate(before=self.timespan[0], after=self.timespan[1])
        return df

    def _fillna(self, df):
        if self.fillna:
            df[self.variable] = fillna(df, self.variable, self.fillna_window_size)
        return df

    def _smooth(self, df):
        if self.smoothing:
            df[self.variable] = smooth(df, self.variable, self.smooth_window_size)
        return df


class Aggregator(ABC):
    """
    An abstract class for aggregating time series data based on different time steps.

    Attributes:

    df: pd.DataFrame
        The input DataFrame containing the time series data to be aggregated.

    variable: str
        The variable/column in the DataFrame to be aggregated.

    fillna: bool
        Fill NaN values in the time series data using a moving window average.

    fillna_window_size: int

    smoothing: bool
        Smooth the time series data using a moving window average.

    smooth_window_size: int
        The size of the moving window for smoothing(n-days). It is recommended to be an odd number.

    timespan: list[str, str] optional
        The start and end dates for a timespan to be aggregated. Format: ['YYYY-MM-DD ]

    agg_metric: str
        The aggregation metric to be used. Supported values: 'mean', 'median', 'min', 'max', 'std', etc.

    Methods:
    --------

    aggregate:
        Aggregates the time series data based on the provided time step.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        variable: str,
        fillna: bool = False,
        fillna_window_size: int = None,
        smoothing: bool = False,
        smooth_window_size: int = None,
        timespan: List[str] = None,
        agg_metric: str = "mean",
    ):
        self.original_df = df
        self.var = variable
        self.fillna = fillna
        self.fillna_window_size = fillna_window_size
        self.smoothing = smoothing
        self.smooth_window_size = smooth_window_size
        self.timespan = timespan
        self.agg_metric = agg_metric

        self.preprocessor = Preprocessor(
            df,
            variable,
            fillna,
            fillna_window_size,
            smoothing,
            smooth_window_size,
            timespan,
        )
        self.resulted_df = pd.DataFrame()

    @abstractmethod
    def aggregate(self, **kwargs):
        pass

    @property
    def preprocess_df(self):
        return self.preprocessor.preprocess()


class MonthlyAggregator(Aggregator):
    """
    Aggregates the time series data based on month-based time step.
    """

    def aggregate(self, **kwargs):

        self.resulted_df[f"{self.var}-{self.agg_metric}"] = monthly_agg(
            self.preprocess_df, self.var, self.agg_metric
        )

        return filter_df(self.resulted_df, **kwargs)


class DekadalAggregator(Aggregator):
    """
    Aggregates the data based on dekad-based time step.
    """

    def aggregate(self, **kwargs):

        self.resulted_df[f"{self.var}-{self.agg_metric}"] = dekadal_agg(
            self.preprocess_df, self.var, self.agg_metric
        )

        return filter_df(self.resulted_df, **kwargs)


class WeeklyAggregator(Aggregator):
    """
    Aggregates the time series data based on week-based time step.
    """

    def aggregate(self, **kwargs):

        self.resulted_df[f"{self.var}-{self.agg_metric}"] = weekly_agg(
            self.preprocess_df, self.var, self.agg_metric
        )

        return filter_df(self.resulted_df, **kwargs)


class BimonthlyAggregator(Aggregator):
    """
    Aggregates the time series data based on bimonthly (twice a month) time step.
    """

    def aggregate(self, **kwargs):

        self.resulted_df[f"{self.var}-{self.agg_metric}"] = bimonthly_agg(
            self.preprocess_df, self.var, self.agg_metric
        )

        return filter_df(self.resulted_df, **kwargs)


class DailyAggregator(Aggregator):
    """
    Aggregates the time series data based on daily time step.
    """

    def aggregate(self, **kwargs):
        self.resulted_df[f"{self.var}-{self.agg_metric}"] = self.preprocess_df[self.var]
        return filter_df(self.resulted_df, **kwargs)


AGGREGATOR_MAPPING = {
    "month": MonthlyAggregator,
    "dekad": DekadalAggregator,
    "week": WeeklyAggregator,
    "bimonth": BimonthlyAggregator,
    "day": DailyAggregator,
}


class Validator:
    """
    A class for validating the input parameters for the climatology computation.

    Methods:
    --------

    validate:
        Validates the input parameters for the climatology computation.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        variable: str,
        fillna: bool = False,
        fillna_window_size: int = None,
        smoothing: bool = False,
        smooth_window_size: int = None,
        time_step: str = None,
        metrics: List[str] = None,
        time_span: List[str] = None,
    ):
        self.df = df
        self.variable = variable
        self.fillna = fillna
        self.fillna_window_size = fillna_window_size
        self.smoothing = smoothing
        self.smooth_window_size = smooth_window_size
        self.time_step = time_step
        self.metrics = metrics
        self.timespan = time_span

    def validate(self):
        self._validate_df_index()
        self._validate_variable()
        self._validate_fillna_smoothing()
        self._validate_timespan(self.timespan)
        self._validate_time_step(self.time_step)
        self._validate_normal_metrics(self.metrics)

    def _validate_df_index(self):
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("df index must be a datetime index")

    def _validate_variable(self):
        if self.variable not in self.df.columns:
            raise ValueError(
                f"Variable '{self.variable}' not found in the input DataFrame columns."
            )

    def _validate_fillna_smoothing(self):
        if any(
            [
                self.fillna and self.fillna_window_size is None,
                self.smoothing and self.smooth_window_size is None,
            ]
        ):
            raise ValueError(
                "window size must be provided when 'fillna' or 'smoothing' is enabled"
            )

    def _validate_timespan(self, timespan):
        if timespan and len(timespan) != 2:
            raise ValueError(
                "timespan must be a list containing two dates: ['start_date', 'end_date']"
            )

    def _validate_time_step(self, time_step):
        valid_time_steps = ["month", "dekad", "week", "day", "bimonth"]
        if time_step not in valid_time_steps:
            raise ValueError(
                f"Invalid time step '{time_step}'. Supported values: {valid_time_steps}"
            )

    def _validate_normal_metrics(self, metrics):
        valid_metrics = ["mean", "median", "min", "max", "std"]
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Invalid metric '{metric}'. Supported values: {valid_metrics}"
                )


class Climatology:
    """
    A class for calculating climatology(climate normal) for time series data.

    Attributes:
    -----------
    df_original: pd.DataFrame
        The original input DataFrame before resampling and removing NaN values.

    df: pd.DataFrame
        The input DataFrame containing the preprocessed data to be aggregated.

    variable: str
        The variable/column in the DataFrame to be aggregated.

    fillna: bool
        Fill NaN values in the time series data using a moving window average.

    fillna_window_size: int
        The size of the moving window for filling NaN values. It is recommended to be an odd number.

    smoothing: bool
        Smooth the time series data using a moving window average.

    smooth_window_size: int
        The size of the moving window for smoothing(n-days). It is recommended to be an odd number.

    timespan: list[str, str] optional
        The start and end dates for a timespan to be aggregated. Format: ['YYYY-MM-DD ]

    time_step: str
        The time step for aggregation. Supported values: 'day', 'week', 'dekad', 'bimonth', 'month'.

    agg_metric: str
        The aggregation metric to be used. Supported values: 'mean', 'median', 'min', 'max', 'std', etc.

    normal_metrics: List[str]
        The metrics to be used in the climatology computation. Supported values: 'mean', 'median', 'min', 'max', etc.

    clima_df: pd.DataFrame
        The DataFrame containing climatology information.

    Methods:
    --------

    compute_normals:
        Calculates climatology based on the aggregated data.

    plot_ts:
        Plot the time series data for the provided dataframe.
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
        Initializes the Climatology class.
        """
        self.validator = Validator(
            df,
            variable,
            fillna,
            fillna_window_size,
            smoothing,
            smooth_window_size,
            time_step,
            normal_metrics,
            timespan,
        )

        self.validator.validate()
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
        self.clim_df = pd.DataFrame()
        self.aggregated_df = self._perform_aggregation()

    def _perform_aggregation(self):

        params = {
            "df": self.df,
            "variable": self.var,
            "fillna": self.fillna,
            "fillna_window_size": self.fillna_window_size,
            "smoothing": self.smoothing,
            "smooth_window_size": self.smooth_window_size,
            "timespan": self.timespan,
            "agg_metric": self.agg_metric,
        }
        aggregator_class = AGGREGATOR_MAPPING.get(self.time_step)
        return aggregator_class(**params).aggregate()

    def compute_normals(self, **kwargs) -> pd.DataFrame:
        """
        Calculates climatology based on the aggregated data.

        Parameters:
        -----------
        kwargs:
            Additional time/date filtering parameters.

        Returns:
        --------
        pd.DataFrame
            The DataFrame containing climatology information.
        """

        self.clim_df = compute_clim(
            self.aggregated_df,
            self.time_step,
            f"{self.var}-{self.agg_metric}",
            self.normal_metrics,
        )

        return filter_df(self.clim_df, **kwargs)


if __name__ == "__main__":
    ascat_ds = "/home/m294/ascat_dataset"
    from smadi.data_reader import extract_obs_ts

    # sample point in Germany

    lat = 51.0
    lon = 10.0
    raw_df = extract_obs_ts((lon, lat), ascat_ds, obs_type="sm")
    clim = Climatology(
        raw_df,
        "sm",
        fillna=True,
        fillna_window_size=7,
        smoothing=True,
        smooth_window_size=7,
        time_step="month",
        normal_metrics=["mean", "std"],
        agg_metric="mean",
    )
    df = clim.compute_normals()
    from smadi.preprocess import validate_distribution

    l = df["sm-mean"].values.tolist()
    print(l)
    validate_distribution(l)
