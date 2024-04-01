"""
run_workflow.py - SMADI Workflow Execution

"""

__author__ = "Muhammed Abdelaal"
__email__ = "muhammedaabdelaal@gmail.com"

from argparse import ArgumentParser, Namespace, ArgumentError
from typing import List, Tuple, Union, Dict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd
import numpy as np


from smadi.metadata import _Detectors
from smadi.data_reader import AscatData
from smadi.utils import (
    create_logger,
    log_exception,
    log_time,
    load_gpis_by_country,
    get_gpis_from_bbox,
)


def setup_argument_parser() -> ArgumentParser:
    """
    Setup argument parser for SMADI workflow execution.
    """
    parser = ArgumentParser(
        description="Run the SMADI workflow for anomaly detection in ASCAT data"
    )

    # Required arguments
    parser.add_argument(
        "data_path", metavar="data_path", type=str, help="Path to the ASCAT data"
    )
    parser.add_argument(
        "aoi",
        metavar="aoi",
        type=str,
        help="Country name or bounding box coordinates\
        in the tuple 'lon_min, lon_max , lat_min, lat_max'",
    )
    parser.add_argument(
        "time_step",
        metavar="time_step",
        type=str,
        default="month",
        choices=["month", "dekad", "week", "bimonth", "day"],
        help="The time step for the climatology calculation. Supported values: month, dekad, week, bimonth, day",
    )

    # Optional arguments
    parser.add_argument(
        "--variable",
        metavar="variable",
        type=str,
        default="sm",
        help="The variable to be used for the anomaly detection.",
    )
    parser.add_argument(
        "--year",
        metavar="year",
        type=int,
        nargs="*",
        default=None,
        required=True,
        choices=range(2007, 2023),
        help="The year(s) for the date parameters",
    )
    parser.add_argument(
        "--month",
        metavar="month",
        type=int,
        nargs="*",
        default=None,
        choices=range(1, 13),
        help="The month(s) for the date parameters",
    )
    parser.add_argument(
        "--dekad",
        metavar="dekad",
        type=int,
        nargs="*",
        default=None,
        choices=(1, 2, 3),
        help="The dekad(s) for the date parameters",
    )
    parser.add_argument(
        "--week",
        metavar="week",
        type=int,
        nargs="*",
        default=None,
        choices=range(1, 53),
        help="The week(s) for the date parameters",
    )
    parser.add_argument(
        "--bimonth",
        metavar="bimonth",
        type=int,
        nargs="*",
        default=None,
        choices=(1, 2),
        help="The bimonth(s) for the date parameters",
    )
    parser.add_argument(
        "--day",
        metavar="day",
        type=int,
        nargs="*",
        default=None,
        choices=range(1, 32),
        help="The day(s) for the date parameters",
    )
    parser.add_argument(
        "--methods",
        metavar="methods",
        type=str,
        nargs="*",
        default=["zscore"],
        help="Anomaly detection methods. Supported methods: zscore, smapi-mean,\
            smapi-median, smdi, smca-mean, smca-median, smad, smci, smds, essmi, beta, gamma",
    )
    parser.add_argument(
        "--timespan",
        metavar="timespan",
        type=list,
        default=None,
        help="To work on a subset of the data. Example: ['2012-01-01', '2012-12-31']",
    )
    parser.add_argument(
        "--fillna", type=bool, default=False, help="Fill missing values"
    )
    parser.add_argument("--fillna_size", type=int, default=3, help="Fillna window size")
    parser.add_argument("--smoothing", type=bool, default=False, help="Apply smoothing")
    parser.add_argument(
        "--smooth_size", type=int, default=31, help="Smoothing window size"
    )
    parser.add_argument(
        "--workers",
        metavar="workers",
        type=int,
        default=None,
        help="The number of workers to use for multiprocessing",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="Save the output to a file to the given path",
    )

    return parser


parser = setup_argument_parser()
args: Namespace = parser.parse_args()

# Required parameters
data_path = args.data_path
aoi = args.aoi
if "," in aoi:
    aoi = tuple(map(float, aoi.split(",")))
variable = args.variable
time_step = args.time_step

# Optional parameters
methods = args.methods
workers = args.workers
fillna = args.fillna
fillna_window_size = args.fillna_size
smoothing = args.smoothing
smooth_window_size = args.smooth_size
timespan = args.timespan
year = args.year
month = args.month
dekad = args.dekad
week = args.week
bimonth = args.bimonth
day = args.day
save_to = args.save_to

# Create an instance of the AscatData class
ascat_obj = AscatData(data_path, False)

# Create a logger
logger = create_logger("run_logger")


@log_exception(logger)
def load_ts(gpi, variable="sm"):
    """
    Load ASCAT time series for a given gpi
    """
    ascat_ts = ascat_obj.read(gpi)
    valid = ascat_ts["num_sigma"] >= 2
    ascat_ts.loc[~valid, ["sm", "sigma40", "slope40", "curvature40"]] = np.nan
    df = pd.DataFrame(ascat_ts.get(variable))
    return df


def _validate_param(param_name: str, param_value: List[int]) -> None:

    if param_value is None:
        raise ValueError(f"The '{param_name}' parameter must be provided")
    if not (isinstance(param_value, (list, int))):
        raise ValueError(f"The '{param_name}' parameter must be an int of list of ints")


def _validate_required_params(
    time_step: str,
    required_params: Dict[str, List[str]],
    local_vars: Dict[str, List[int]],
) -> None:

    missing_params = [
        param for param in required_params[time_step] if local_vars.get(param) is None
    ]
    if missing_params:
        raise ValueError(
            f"For time_step '{time_step}', the following parameters must be provided: "
            f"{', '.join(missing_params)}"
        )


def validate_date_params(
    time_step: str,
    year: Union[int, List[int]] = None,
    month: Union[int, List[int]] = None,
    dekad: Union[int, List[int]] = None,
    week: Union[int, List[int]] = None,
    bimonth: Union[int, List[int]] = None,
    day: Union[int, List[int]] = None,
) -> Dict[str, List[int]]:
    """
    Validate the date parameters for the anomaly detection workflow.
    """

    year = [year] if isinstance(year, int) else year
    month = [month] if isinstance(month, int) else month
    dekad = [dekad] if isinstance(dekad, int) else dekad
    week = [week] if isinstance(week, int) else week
    bimonth = [bimonth] if isinstance(bimonth, int) else bimonth
    day = [day] if isinstance(day, int) else day

    date_param = {"year": year, "month": month}

    # Validating dekad or week based on time_step

    if time_step == "month":
        pass
    elif time_step == "dekad":
        date_param["dekad"] = dekad
    elif time_step == "bimonth":
        date_param["bimonth"] = bimonth

    elif time_step == "day":
        date_param["day"] = day

    elif time_step == "week":
        date_param["week"] = week
        date_param.pop("month")

    else:
        raise ValueError(
            f"Unsupported time_step: {time_step}. Supported time_steps are month, dekad, week, bimonth, day"
        )

    # Validation for parameters
    for param_name, param_value in date_param.items():
        _validate_param(param_name, param_value)

    # Checking if the value lists are of the same length
    if len(set(map(len, date_param.values()))) > 1:
        raise ValueError(
            "The length of the date parameters lists must be the same for multiple dates"
        )

    # Checking if required parameters are provided
    required_params = {
        "month": ["year", "month"],
        "dekad": ["year", "month", "dekad"],
        "week": ["year", "week"],
        "bimonth": ["year", "month", "bimonth"],
        "day": ["year", "month", "day"],
    }

    local_vars = locals()
    _validate_required_params(time_step, required_params, local_vars)

    return date_param


def validate_anomaly_method(methods):
    """
    Validate the anomaly detection method
    """

    for method in methods:
        if method not in _Detectors.keys():
            raise ValueError(
                f"Anomaly method '{method}' is not supported."
                f"Supported methods are one of the following: {tuple(_Detectors.keys())}"
            )


@log_exception(logger)
def single_po_run(
    gpi: int,
    methods: str = ["zscore"],
    variable: str = "sm",
    time_step: str = "month",
    fillna: bool = False,
    fillna_window_size: int = None,
    smoothing: bool = False,
    smooth_window_size: int = None,
    year: Union[int, List[int]] = None,
    month: Union[int, List[int]] = None,
    dekad: Union[int, List[int]] = None,
    week: Union[int, List[int]] = None,
    bimonth: Union[int, List[int]] = None,
    day: Union[int, List[int]] = None,
    timespan: List[str] = None,
) -> Tuple[int, Dict[str, float]]:
    """
    Run the anomaly detection workflow for a single grid point index.
    """

    # Load the time series for the given gpi
    global ascat_obj
    df = load_ts(gpi, variable=variable)
    # Validate the date parameters
    date_params = validate_date_params(
        time_step, year, month, dekad, week, bimonth, day
    )
    # Create a list of dictionaries containing the date parameters
    date_params = [
        dict(zip(date_params.keys(), values)) for values in zip(*date_params.values())
    ]

    # Define a dictionary to store the results
    results = {}
    for method in methods:

        # Define the anomaly detection parameters
        anomaly_params = {
            "df": df,
            "variable": variable,
            "time_step": time_step,
            "fillna": fillna,
            "fillna_window_size": fillna_window_size,
            "smoothing": smoothing,
            "smooth_window_size": smooth_window_size,
            "timespan": timespan,
        }

        # If the method has a metric parameter (e.g. smapi-mean, smapi-median), set the metric parameter
        if "-" in method:
            anomaly_params["normal_metrics"] = [method.split("-")[1]]

        elif method in ["beta", "gamma"]:
            anomaly_params["dist"] = [method]

        try:
            for date_param in date_params:
                anomaly = _Detectors[method](**anomaly_params).detect_anomaly(
                    **date_param
                )
                date_str = f"-".join(str(value) for value in date_param.values())
                results[method + f"({date_str})"] = anomaly[method].values[0]

        except AttributeError as e:
            return None

    return (gpi, results)


@log_exception(logger)
def _finalize(result: Tuple[int, dict], df: pd.DataFrame, gpis_col="point"):
    try:
        gpi, anomaly = result
    except Exception as e:
        return df

    else:
        for method, value in anomaly.items():
            df.loc[df[gpis_col] == gpi, method] = value

    return df


@log_time(logger)
def run(
    aoi: Union[str, Tuple[float, float, float, float]],
    methods: Union[str, List[str]] = ["zscore"],
    variable: str = "sm",
    time_step: str = "month",
    fillna: bool = False,
    fillna_window_size: int = None,
    smoothing: bool = False,
    smooth_window_size: int = None,
    timespan: List[str] = None,
    year: List[int] = None,
    month: List[int] = None,
    dekad: List[int] = None,
    week: List[int] = None,
    bimonth: List[int] = None,
    day: List[int] = None,
    workers: int = None,
) -> pd.DataFrame:
    """
    Run the anomaly detection workflow for multiple grid point indices with multiprocessing support.
    """
    # Print workflow start message
    print("\nWorkflow started....\n")
    print(f"Loading grid points for {aoi}....")

    if isinstance(aoi, str):
        pointlist = load_gpis_by_country(aoi)
    else:
        pointlist = get_gpis_from_bbox(aoi)

    print(f"Grid points loaded successfully for {aoi}\n")
    print(pointlist.head())
    print("\n")
    pointlist = pointlist[:100]
    pre_compute = partial(
        single_po_run,
        methods=methods,
        variable=variable,
        time_step=time_step,
        fillna=fillna,
        fillna_window_size=fillna_window_size,
        smoothing=smoothing,
        smooth_window_size=smooth_window_size,
        year=year,
        month=month,
        dekad=dekad,
        week=week,
        bimonth=bimonth,
        day=day,
        timespan=timespan,
    )

    # Print the workflow initiation message
    print(f"Running the anomaly detection workflow for {aoi}....\n")

    # Print workflow parameters
    print("Workflow parameters:\n")
    print(f"    Anomaly detection methods: {', '.join(methods)}")
    print(f"    Variable: {variable}")
    print(f"    Time step for Climatology: {time_step}")
    print(f"    Date parameters:\n")

    # Print date parameters if available
    date_parameters = {
        "Year": year,
        "Month": month,
        "Dekad": dekad,
        "Week": week,
        "Bimonth": bimonth,
        "Day": day,
    }
    for param, value in date_parameters.items():
        if value:
            print(f"     {param}: {value}")

    with ProcessPoolExecutor(workers) as executor:
        results = executor.map(pre_compute, pointlist["point"])
        for result in results:
            pointlist = _finalize(result, pointlist)

        return pointlist


def main():

    try:
        validate_anomaly_method(methods)
        validate_date_params(time_step, year, month, dekad, week, bimonth, day)
    except ArgumentError as e:
        parser.error(str(e))

    from time import time

    start = time()
    df = run(
        aoi=aoi,
        methods=methods,
        variable=variable,
        time_step=time_step,
        year=year,
        month=month,
        week=week,
        day=day,
        bimonth=bimonth,
        dekad=dekad,
        timespan=timespan,
        fillna=fillna,
        fillna_window_size=fillna_window_size,
        smoothing=smoothing,
        smooth_window_size=smooth_window_size,
        workers=workers,
    )

    print(f"\nWorkflow completed in {time() - start} seconds\n")
    print(df)

    if save_to:
        try:
            df.to_csv(save_to)
            print(f"Saving the output data frame to {save_to}....")
            print("Output saved successfully")

        except ArgumentError as e:
            parser.error(str(e))


if __name__ == "__main__":
    main()
