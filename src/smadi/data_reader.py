import os
import numpy as np
import pandas as pd
import xarray as xr
from fibgrid.realization import FibGrid
from pynetcf.time_series import GriddedNcContiguousRaggedTs
from pynetcf.time_series import GriddedNcTs
from pygeogrids.netcdf import load_grid
from pynetcf.time_series import OrthoMultiTs


class AscatData(GriddedNcContiguousRaggedTs):
    """
    Class reading ASCAT SSM 6.25 km data.
    """

    def __init__(self, path, read_bulk=True):
        """
        Initialize ASCAT data.

        Parameters
        ----------
        path : str
            Path to dataset.
        read_bulk : bool, optional
            If "True" all data will be read in memory, if "False"
            only a single time series is read (default: False).
            Use "True" to process multiple GPIs in a loop and "False" to
            read/analyze a single time series.
        """
        grid = FibGrid(6.25)
        ioclass_kws = dict(read_bulk=read_bulk)
        super().__init__(path, grid, ioclass_kws=ioclass_kws)


class Era5Land(GriddedNcTs):
    """
    Read time series data from ERA5 netCDF files.
    """

    def __init__(self, path, read_bulk=True, celsius=True):
        """
        Parameters
        ----------
        path : str
            Path to the data.
        read_bulk : boolean, optional
            If "True" all data will be read in memory, if "False"
            only a single time series is read (default: False).
            Use "True" to process multiple GPIs in a loop and "False" to
            read/analyze a single time series.
        celsius: boolean, optional
            if True temperature values are returned in degrees Celsius,
            otherwise they are in degrees Kelvin
            Default : True
        """
        parameter = ["swvl1", "stl1", "tp", "sd"]

        grid_filename = os.path.join(path, "grid.nc")
        grid = load_grid(grid_filename)

        if type(parameter) != list:
            parameter = [parameter]

        self.parameters = parameter

        offsets = {}
        self.path = {}

        self.path["ts"] = path

        param_list = ["139", "167", "170", "183", "235", "236", "stl1", "2t", "t2m"]

        for parameter in self.parameters:
            if celsius and parameter in param_list:
                offsets[parameter] = -273.15
            else:
                offsets[parameter] = 0.0

        super(Era5Land, self).__init__(
            self.path["ts"],
            ioclass=OrthoMultiTs,
            grid=grid,
            ioclass_kws={"read_bulk": read_bulk},
            parameters=self.parameters,
            offsets=offsets,
        )


def read_grid_point(loc, ascat_sm_path, era5_land_path=None, read_bulk=False):
    """
    Read grid point for given lon/lat coordinates or grid_point.

    Parameters
    ----------
    loc : int, tuple
        Tuple is interpreted as longitude, latitude coordinate.
        Integer is interpreted as grid point index.

    ascat_sm_path : str
        Path to ASCAT soil moisture data.

    era5_land_path : str
        Path to ERA5-Land data.

    read_bulk : bool, optional
        If "True" all data will be read in memory, if "False"
        only a single time series is read (default: False).
        Use "True" to process multiple GPIs in a loop and "False" to
        read/analyze a single time series.
    """
    data = {}

    print(f"Reading ASCAT soil moisture: {ascat_sm_path}")
    ascat_obj = AscatData(ascat_sm_path, read_bulk)

    if isinstance(loc, tuple):
        lon, lat = loc
        ascat_gpi, distance = ascat_obj.grid.find_nearest_gpi(lon, lat)
        print(f"ASCAT GPI: {ascat_gpi} - distance: {distance:8.3f} m")
    else:
        ascat_gpi = loc
        lon, lat = ascat_obj.grid.gpi2lonlat(ascat_gpi)
        print(f"ASCAT GPI: {ascat_gpi}")

    ascat_ts = ascat_obj.read(ascat_gpi)

    if ascat_ts is None:
        raise RuntimeError(f"ASCAT soil moisture data not found: {ascat_sm_path}")

    # set observations to NaN with less then two observations
    valid = ascat_ts["num_sigma"] >= 2
    ascat_ts.loc[~valid, ["sm", "sigma40", "slope40", "curvature40"]] = np.nan
    data["ascat_ts"] = ascat_ts
    data["ascat_gpi"] = ascat_gpi
    data["ascat_lon"] = lon
    data["ascat_lat"] = lat

    if era5_land_path is not None:
        print(f"Reading ERA5-Land: {era5_land_path}")
        era5_land_obj = Era5Land(era5_land_path, read_bulk)
        era5_land_gpi, distance = era5_land_obj.grid.find_nearest_gpi(lon, lat)
        era5_land_lon, era5_land_lat = era5_land_obj.grid.gpi2lonlat(era5_land_gpi)
        era5_land_ts = era5_land_obj.read(era5_land_gpi)
        print(f"ERA5-Land GPI: {era5_land_gpi} - distance: {distance:8.3f} m")
    else:
        era5_land_ts = None
        era5_land_gpi = None
        era5_land_lon = None
        era5_land_lat = None
        print(f"Warning: ERA5-Land not found: {era5_land_path}")

    data["era5_land_ts"] = era5_land_ts
    data["era5_land_gpi"] = era5_land_gpi
    data["era5_land_lon"] = era5_land_lon
    data["era5_land_lat"] = era5_land_lat

    if era5_land_ts is not None:

        ts = pd.merge_asof(
            data["ascat_ts"],
            data["era5_ts"],
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta("3h"),
            direction="nearest",
        )

        # mask data that is either frozen (temperature below 0) or with snow
        not_valid = (ts["stl1"] < 0) | (ts["sd"] > 0)
        data["ascat_ts"]["sm_valid"] = ~not_valid
    else:
        data["ascat_ts"]["sm_valid"] = True
        print("Warning: ERA5 Land not found - ASCAT soil moisture not masked!")

    return data


def extract_obs_ts(
    loc, ascat_path, era5_land_path=None, obs_type="sm", read_bulk=False
):
    """
    Read time series of given observation type.

    Parameters
    ----------
    loc : int, tuple
        Tuple is interpreted as longitude, latitude coordinate.
        Integer is interpreted as grid point index.
    ascat_path : str
        Path to ASCAT soil moisture data.

    era5_land_path : str
        Path to ERA5-Land data.

    obs : str, optional
        Observation type (default: "sm").
    read_bulk : bool, optional
        If "True" all data will be read in memory, if "False"
        only a single time series is read (default: False).
        Use "True" to process multiple GPIs in a loop and "False" to
        read/analyze a single time series.
    """
    data = read_grid_point(loc, ascat_path, era5_land_path, read_bulk)
    ascat_ts = data.get("ascat_ts")
    # lat = data.get("ascat_lat")
    # lon = data.get("ascat_lon")
    # gpi = data.get("ascat_gpi")
    ts = ascat_ts.get(obs_type)
    ts.dropna(inplace=True)

    # return {"ts": pd.DataFrame(ts), "lon": lon, "lat": lat, "gpi": gpi}

    return pd.DataFrame(ts)


def read_era5(era5_path, loc, interpo_method="nearest"):
    """
    Read ERA5 data for given location.

    Parameters
    ----------

    era5_path : str
        Path to ERA5 dataset.

    loc : tuple
        Tuple is interpreted as longitude, latitude coordinate.

    interpo_method : str, optional
        Interpolation method (default: "nearest").

    Returns
    -------

    ts : pd.DataFrame
        Time series of ERA5 data containing all variables.
    """
    ds = xr.open_dataset(era5_path)
    ds_point = ds.sel(lat=loc[1], lon=loc[0], method=interpo_method)
    ts = ds_point.to_dataframe()
    ts = ts.drop(columns=["lat", "lon"])
    ts.index.name = None

    return ts
