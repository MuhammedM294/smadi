import eomaps
from smadi.metadata import indicators_thresholds


def set_thresholds(method):
    """
    Set the thresholds for the specified method based on the method name.

    parameters:
    -----------

    method: str
        The method name for which the thresholds are to be set. Supported methods are:
        'zscore', 'smapi', 'smdi', 'smca', 'smad', 'smci', 'smds', 'essmi', 'beta', 'gamma'
    """

    if method in indicators_thresholds.keys():
        return indicators_thresholds[method]
    else:
        return None


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
    crs=4326,
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
    maps_titles_kwargs={"pad": 20, "fontsize": 12, "fontweight": "bold"},
    add_features=True,
    frame_line_width=1,
    cmap="RdYlBu",
    add_gridlines=False,
    cb_kwargs={
        "pos": 0.4,
        "labelsize": 0.5,
        "tick_lines": "center",
        "show_values": False,
    },
):
    m = eomaps.Maps(ax=(ax_rows, ax_cols, 1), figsize=figsize)
    figure_title_kwargs["s"] = figure_title
    m.text(**figure_title_kwargs)

    for i, colm in enumerate(df_colms):
        ax_index = i + 1
        if i == 0:
            m = m
        else:
            m = m.new_map(ax=(ax_rows, ax_cols, ax_index), figsize=(7, 7))
        m.ax.set_title(maps_titles[i], **maps_titles_kwargs)
        m.set_shape.rectangles(radius=0.05)
        if add_features:
            m.add_feature.preset.coastline(lw=0.6)
            m.add_feature.preset.countries(lw=0.4, ls="--")
            m.add_feature.preset.ocean()
            m.add_feature.preset.land()

        m.set_frame(linewidth=frame_line_width)
        m.set_data(data=df, parameter=colm, x=x, y=y, crs=crs)

        if add_gridlines:
            g = m.add_gridlines(
                d=(2, 2),
                ec="grey",
                ls="--",
                lw=0.01,
            )
            g.add_labels(fontsize=8)
        clss = set_bins(colm)
        bins = clss[0] if clss else [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        labels = (
            clss[1]
            if clss
            else [
                "0-10",
                "10-20",
                "20-30",
                "30-40",
                "40-50",
                "50-60",
                "60-70",
                "70-80",
                "80-90",
                "90-100",
            ]
        )
        vmin = clss[0][0] if clss else df[colm].min()
        vmax = clss[0][-1] if clss else df[colm].max()

        m.set_classify.UserDefined(bins=bins)

        if colm.split("(")[0] == "smds":
            cmap = cmap + "_r"
        m.plot_map(vmin=vmin, vmax=vmax, cmap=cmap, lw=1.5)
        cb = m.add_colorbar(
            label=False, spacing="uniform", pos=0.4, orientation="vertical"
        )
        cb.set_hist_size(0.8)
        cb.tick_params(rotation=0, labelsize=10, pad=5)

        if clss:
            bins, labels = clss
            cb.set_bin_labels(
                bins=bins,
                names=labels,
                tick_lines=cb_kwargs["tick_lines"],
                show_values=cb_kwargs["show_values"],
            )

    m.show()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("test1.csv")
    df.dropna(inplace=True)

    parameters = [
        "sm-avg(2021-7)",
        "norm-mean(2021-7)",
        "abs-mean(2021-7)",
        "abs-median(2021-7)",
        "zscore(2021-7)",
        "smad(2021-7)",
    ]
    titles = [
        "SSM Average ",
        "Climatology",
        "Absolute Anomaly- Mean",
        "Absolute Anomaly- Median",
        "Z-Score",
        "SMAD",
    ]
    plot_anomaly_maps(
        figsize=(25, 20),
        ax_cols=3,
        ax_rows=2,
        df=df,
        df_colms=parameters,
        figure_title="ASCAT SSM Anomaly Maps for Germany, July 2021",
        maps_titles=titles,
        add_gridlines=False,
        add_features=True,
    )
