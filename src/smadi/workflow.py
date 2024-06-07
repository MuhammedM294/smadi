"""
run_workflow.py - SMADI Workflow Execution

"""

from argparse import ArgumentParser, Namespace, ArgumentError
from typing import List, Tuple, Union, Dict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd
import numpy as np
from tqdm import tqdm


from smadi.data_reader import AscatData
from smadi.anomaly_detectors import _Detectors
from smadi.preprocess import validate_date_params, validate_anomaly_method, filter_df
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
        description="Run the SMADI workflow for anomaly detection on ASCAT data"
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
        in str format 'lon_min, lon_max , lat_min, lat_max'",
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
        "--data_read_bulk",
        type=bool,
        default=False,
        help="Read data in bulk mode. If 'True' all data will be read in memory",
    )
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
            smapi-median, smdi, smca-mean, smca-median, smad, smci, smds, essmi,\
                beta, gamma",
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
    parser.add_argument(
        "--fillna_window_size", type=int, default=3, help="Fillna window size"
    )
    parser.add_argument("--smoothing", type=bool, default=False, help="Apply smoothing")
    parser.add_argument(
        "--smooth_window_size", type=int, default=31, help="Smoothing window size"
    )
    parser.add_argument(
        "--workers",
        metavar="workers",
        type=int,
        default=None,
        help="The number of workers to determine the degree of concurrent processing in a multiprocessing setup",
    )

    parser.add_argument(
        "--addi_retrive",
        metavar="addi_retrive",
        type=bool,
        default=False,
        help="Retrieve observations and  climatology values",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="Save the output to a file to the given path",
    )

    return parser


def parse_arguments(parser):
    """
    Parse the arguments and return the parsed arguments as a dictionary.

    returns:
    --------
    parsed_args: dict
        The parsed arguments as a dictionary
    """
    args: Namespace = parser.parse_args()

    aoi = args.aoi
    if "," in aoi:
        aoi = tuple(map(float, aoi.split(",")))

    parsed_args = vars(args)
    parsed_args["aoi"] = aoi
    return parsed_args


# Create a logger
logger = create_logger("smadi-logger")


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
    addi_retrive: bool = False,
    agg_metric: list = "mean",
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
            "agg_metric": agg_metric,
        }

        # If the method has a metric parameter (e.g. smapi-mean, smapi-median), set the metric parameter
        if "-" in method:
            anomaly_params["normal_metrics"] = [method.split("-")[1]]

        elif method in ["beta", "gamma"]:
            anomaly_params["dist"] = [method]

        try:

            anomaly = _Detectors[method](**anomaly_params).detect_anomaly()
            for date_param in date_params:
                anomaly_df = filter_df(anomaly, **date_param)
                date_str = f"-".join(str(value) for value in date_param.values())
                results[method + f"({date_str})"] = anomaly_df[method].values

                if addi_retrive:

                    results[f"{variable}-{agg_metric}" + f"({date_str})"] = anomaly_df[
                        f"{variable}-{agg_metric}"
                    ].values
                    if "norm-mean" in anomaly.columns:
                        results["norm-mean" + f"({date_str})"] = anomaly_df[
                            "norm-mean"
                        ].values
                    if "norm-median" in anomaly.columns:
                        results["norm-median" + f"({date_str})"] = anomaly_df[
                            "norm-median"
                        ].values

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
def run_smadi(
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
    addi_retrive: bool = False,
) -> pd.DataFrame:
    """
    Run the anomaly detection workflow for multiple grid point indices with multiprocessing support.
    """
    logger.info("Workflow started....\n")
    logger.info(f"Loading grid points for {aoi}....")

    if isinstance(aoi, str):
        pointlist = load_gpis_by_country(aoi)
    else:
        pointlist = get_gpis_from_bbox(aoi)

    logger.info(f"Grid points loaded successfully for {aoi}\n")
    logger.info(f"\n\n{pointlist.head()}")
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
        addi_retrive=addi_retrive,
    )

    logger.info(f"Running the anomaly detection workflow for {aoi}....\n")
    logger.info("Workflow parameters:\n")
    logger.info(f"    Anomaly detection methods: {', '.join(methods)}")
    logger.info(f"    Variable: {variable}")
    logger.info(f"    Time step for Climatology: {time_step}")
    logger.info(f"    Date parameters:\n")

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
            logger.info(f"     {param}: {value}")

    logger.info("Reading ASCAT Data...")
    with ProcessPoolExecutor(workers) as executor:
        results = list(
            tqdm(executor.map(pre_compute, pointlist["point"]), total=len(pointlist))
        )
        for result in results:
            pointlist = _finalize(result, pointlist)

        return pointlist


def main():

    parser = setup_argument_parser()
    args = parse_arguments(parser)
    global ascat_obj
    logger.info("Initializing ASCAT data reader...")
    ascat_obj = AscatData(args["data_path"], args["data_read_bulk"])

    try:
        logger.info("Validating input parameters...")
        validate_anomaly_method(args["methods"], _Detectors)
        validate_date_params(
            args["time_step"],
            args["year"],
            args["month"],
            args["dekad"],
            args["week"],
            args["bimonth"],
            args["day"],
        )
    except ArgumentError as e:
        parser.error(str(e))

    # remove "data_path" from the arguments
    temp_args = args.copy()
    temp_args.pop("data_path")
    temp_args.pop("data_read_bulk")
    temp_args.pop("save_to")

    df = run_smadi(**temp_args)

    logger.info(f"\n\n {df}")

    if args["save_to"]:
        try:
            df.to_csv(args["save_to"])
            logger.info(f"Saving the output data frame to {args['save_to']}....")
            logger.info("Output saved successfully")

        except ArgumentError as e:
            parser.error(str(e))


if __name__ == "__main__":
    main()
