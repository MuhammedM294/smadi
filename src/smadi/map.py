import eomaps
import pandas as pd
from tqdm import tqdm

from smadi.plot import set_thresholds


def set_extent(df, x="lon", y="lat", buffer=2):
    """
    Set the extent for the map based on the provided dataframe and buffer.

    parameters:
    -----------

    df: pd.DataFrame
        The dataframe containing the data.

    x: str
        The column name for the x-axis.

    y: str
        The column name for the y-axis.

    buffer: int
        The buffer to be added to the min and max values of the x and y axis.
    """

    min_lat = df[y].min() - buffer
    max_lat = df[y].max() + buffer
    min_lon = df[x].min() - buffer
    max_lon = df[x].max() + buffer

    return min_lon, max_lon, min_lat, max_lat


def set_bins(colm):
    """
    Set the bins and labels for color classification for the selected column.

    parameters:
    -----------

    colm: str
        The data column name for which the bins and labels are to be set.
    """
    method = colm.split("(")[0]
    if "-" in method:
        method = method.split("-")[0]

    thrsholds = set_thresholds(method)
    if not thrsholds:
        return None
    bins = [val[1] for val in thrsholds.values()]
    labels = [key for key in thrsholds.keys()]
    labels.insert(0, labels[0])
    bins.insert(0, next(iter(thrsholds.values()))[0])

    return bins, labels


def plot_anomaly_maps(
    figsize=(25, 20),
    ax_rows=1,
    ax_cols=1,
    df=None,
    x="lon",
    y="lat",
    df_colms=None,
    map_crs=eomaps.Maps.CRS.Robinson(),
    # map_crs=eomaps.Maps.CRS.PlateCarree(),
    # map_crs=eomaps.Maps.CRS.AzimuthalEquidistant(
    #     central_longitude=10, central_latitude=51
    # ),
    figure_title="",
    figure_title_kwargs={
        "x": 0.5,
        "y": 0.95,
        "fontsize": 15,
        "ha": "center",
        "va": "center",
        "fontweight": "bold",
    },
    maps_titles=None,
    maps_titles_kwargs={"pad": 11, "fontsize": 10, "fontweight": "bold"},
    add_features=True,
    frame_line_width=1,
    add_cb=True,
    cb_min_max=["sm_clim", "sm_clim", "abs", "anomaly", "anomaly"],
    cmap="RdYlBu",
    vmin=None,
    vmax=None,
    add_gridlines=False,
    cb_kwargs={
        "pos": 0.4,
        "labelsize": 0.5,
        "tick_lines": "center",
        "show_values": False,
    },
    cb_label=None,
    save_to=None,
):
    m = eomaps.Maps(ax=(ax_rows, ax_cols, 1), figsize=figsize, crs=map_crs)
    figure_title_kwargs["s"] = figure_title
    m.text(**figure_title_kwargs)

    total_iterations = len(df_colms)

    for i, (colm, cb_min_max) in tqdm(
        enumerate(zip(df_colms, cb_min_max)),
        total=total_iterations,
        desc="Processing maps",
    ):
        ax_index = i + 1
        if i == 0:
            m = m
        else:
            m = m.new_map(ax=(ax_rows, ax_cols, ax_index), figsize=(7, 7), crs=map_crs)

        if maps_titles:
            m.ax.set_title(maps_titles[i], **maps_titles_kwargs)

        m.set_shape.rectangles(radius=0.05)
        if add_features:
            m.add_feature.preset.coastline(lw=0.6)
            m.add_feature.preset.countries(lw=0.4, ls="--")
            m.add_feature.preset.ocean()
            m.add_feature.preset.land()

        m.set_frame(linewidth=frame_line_width)
        m.set_data(data=df, parameter=colm, x=x, y=y, crs=4326)

        if add_gridlines:
            g = m.add_gridlines(
                d=(2, 2),
                ec="grey",
                ls="--",
                lw=0.01,
            )
            g.add_labels(fontsize=8)

        if cb_min_max == "anomaly":
            bins, labels = set_bins(colm)
            vmin = bins[0]
            vmax = bins[-1]
            m.set_classify.UserDefined(bins=bins)

        if cb_min_max == "sm_clim":
            vmin = 0
            vmax = 100
        if cb_min_max == "abs":
            vmin = -30 if df[colm].min() > -30 else df[colm].min()
            vmax = 30 if df[colm].max() < 30 else df[colm].max()

        if cb_min_max == "spi":
            vmin = -1
            vmax = 1

        if colm.split("(")[0] == "smds":
            cmap = cmap + "_r"

        m.plot_map(vmin=vmin, vmax=vmax, cmap=cmap, lw=1.5)

        if add_cb:
            label = cb_label[i] if cb_label else None
            cb = m.add_colorbar(
                label=label,
                spacing="uniform",
                pos=0.4,
                orientation="vertical",
                hist_bins=256,
            )
            cb.set_hist_size(0.8)
            cb.tick_params(rotation=0, labelsize=10, pad=5)

            if cb_min_max == "anomaly":
                cb.set_bin_labels(
                    bins=bins,
                    names=labels,
                    tick_lines=cb_kwargs["tick_lines"],
                    show_values=cb_kwargs["show_values"],
                )

    m.show()
    if save_to:
        m.savefig(save_to)
