import xarray as xr
from pathlib import Path
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial


from smadi.data_reader import AscatData, extract_obs_ts
from smadi.utils import load_gpis_by_country
from smadi.metadata import _Detectors
from smadi.workflow import load_ts, _finalize

from smadi.utils import create_logger, log_exception, log_time


spi_logger = create_logger("spi_corr")


ascat_path = Path("/home/m294/VSA/Code/datasets")
spi_path = "/home/m294/Repo/era5/Germany_monthly_spi_gamma.nc"


spi_ds = xr.open_dataset(spi_path)
spi_ds = spi_ds.sel(time=slice("2007-01-01", "2022-12-31"))

ascat_obj = AscatData(ascat_path, read_bulk=False)


@log_exception(spi_logger)
def load_ts(gpi, variable="sm"):
    """
    Load ASCAT time series for a given gpi
    """
    ascat_ts = ascat_obj.read(gpi)
    valid = ascat_ts["num_sigma"] >= 2
    ascat_ts.loc[~valid, ["sm", "sigma40", "slope40", "curvature40"]] = np.nan
    df = pd.DataFrame(ascat_ts.get(variable))
    return df


@log_exception(spi_logger)
def compute_correlation(gpi, lat, lon, variable, method, time_step, spi_ds):

    df = load_ts(gpi, variable)
    spi = spi_ds.sel(lat=lat, lon=lon, method="nearest")

    anomaly_params = {
        "df": df,
        "variable": variable,
        "time_step": time_step,
        "fillna": True,
        "fillna_window_size": 3,
        "smoothing": True,
        "smooth_window_size": 31,
    }

    if "-" in method:
        anomaly_params["normal_metrics"] = [method.split("-")[1]]

    elif method in ["beta", "gamma"]:
        anomaly_params["dist"] = [method]

    anomaly = _Detectors[method](**anomaly_params).detect_anomaly()
    anomaly_data = anomaly[method].values
    spi_data = spi["spi_gamma_01"].values

    cor, p = pearsonr(anomaly_data, spi_data)

    result = {f"{method}_corr": cor, f"{method}_p": p}

    return (gpi, result)


@log_time(spi_logger)
def spi_workflow(
    country,
    spi_path,
    method="zscore",
    time_step="month",
    variable="sm",
):

    pointlist = load_gpis_by_country(country)
    spi_ds = xr.open_dataset(spi_path)
    spi_ds = spi_ds.sel(time=slice("2007-01-01", "2022-12-31"))

    pre_calc = partial(
        compute_correlation,
        variable=variable,
        method=method,
        time_step=time_step,
        spi_ds=spi_ds,
    )

    with ProcessPoolExecutor() as executor:
        results = list(
            executor.map(
                pre_calc,
                pointlist["point"],
                pointlist["lat"],
                pointlist["lon"],
            )
        )
        for result in results:
            pointlist = _finalize(result, pointlist)

        return pointlist


# if __name__ == "__main__":

#     method = "smca-median"
#     df = spi_workflow("Germany", spi_path, method=method)

#     df.to_csv(f"/home/m294/Repo/spi_anomaly_corr/spi_corr_{method}.csv")

#     df = pd.read_csv(f"/home/m294/Repo/spi_anomaly_corr/spi_corr_{method}.csv")
#     print(df)
#     print(df[f"{method}_corr"].describe())
#     print(df[f"{method}_p"].describe())
