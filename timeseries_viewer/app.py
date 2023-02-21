"""
A pyviz panel/holoviews app for visualizing fMRI timeseries.
"""

from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
from typing import Union

import holoviews as hv
import numpy as np
import panel as pn
from bokeh.models import HoverTool
from matplotlib.colors import LinearSegmentedColormap

hv.extension("bokeh")
pn.extension(sizing_mode="stretch_width")

HEIGHT = 300
CMAP = LinearSegmentedColormap.from_list(
    "conmat", ["blue", "cyan", "white", "yellow", "red"]
)


def main(args):
    timeseries = get_timeseries(args.paths[0])

    path_select = pn.widgets.Select(
        name="TS path", value=args.paths[0], options=args.paths
    )

    frame_slider = pn.widgets.IntSlider(
        name="Time",
        value=0,
        start=0,
        end=timeseries.shape[1],
    )
    width_slider = pn.widgets.IntSlider(
        name="Width",
        value=1,
        start=1,
        step=2,
        end=timeseries.shape[1],
    )
    ranges_input = pn.widgets.TextInput(
        name="Timeseries ranges", placeholder="0,1,3-5", value="0"
    )

    def update_end(path: str):
        timeseries = get_timeseries(path)
        frame_slider.end = timeseries.shape[1]
        width_slider.end = timeseries.shape[1]

    pn.bind(update_end, path=path_select)

    carpet = pn.depends(path=path_select)(plot_carpet)

    conmat = pn.depends(path=path_select, frame=frame_slider, width=width_slider)(
        plot_conmat
    )

    window = pn.depends(path=path_select, frame=frame_slider, width=width_slider)(
        plot_window
    )

    lines = pn.depends(path=path_select, ranges=ranges_input)(plot_lines)
    rms = pn.depends(path=path_select)(plot_rms)

    carpet_dmap = hv.DynamicMap(carpet)
    conmat_dmap = hv.DynamicMap(conmat)
    window_dmap = hv.DynamicMap(window)
    lines_dmap = hv.DynamicMap(lines)
    rms_dmap = hv.DynamicMap(rms)

    app = pn.template.BootstrapTemplate(title="Timeseries viewer")
    app.sidebar.extend([path_select, frame_slider, width_slider, ranges_input])
    app.main.append(
        pn.Column(
            ((carpet_dmap * window_dmap) + conmat_dmap).opts(shared_axes=False),
            ((lines_dmap * window_dmap) + (rms_dmap * window_dmap)).cols(1),
        )
    )
    pn.serve(app, port=args.port)


@lru_cache(maxsize=2)
def get_timeseries(path: Union[str, Path], scale: bool = False):
    path = Path(path)

    timeseries: np.ndarray
    if path.suffix == ".tsv":
        # TODO: what is the shape of these files?
        timeseries = np.loadtxt(path, delimiter="\t", dtype=float)
    elif path.suffix == ".npy":
        timeseries = np.load(path)
    else:
        raise ValueError(f"Invalid path {path}; expected .tsv or .npy")

    timeseries = np.ascontiguousarray(timeseries.T)
    timeseries -= timeseries.mean(axis=1, keepdims=True)
    timeseries /= timeseries.std(axis=1, keepdims=True) + 1e-8

    if scale:
        timeseries /= np.sqrt(np.mean(timeseries**2, axis=0))
    return timeseries


def get_window(frame: int, width: int, length: int):
    start = max(frame - width // 2, 0)
    stop = min(start + width, length)
    return start, stop


def plot_conmat(path: str, frame: int, width: int):
    timeseries = get_timeseries(path)

    start, stop = get_window(frame, width, timeseries.shape[1])
    window = timeseries[:, start:stop]
    mat = (1 / window.shape[1]) * window @ window.T
    vmax = np.quantile(np.abs(mat), 0.95)

    # NOTE: found the magic '@image' here:
    # https://github.com/holoviz/holoviews/blob/main/holoviews/plotting/bokeh/raster.py#L44
    tooltips = [
        ("(i, j)", "($y{0}, $x{0})"),
        ("value", "@image{0.00}"),
    ]
    hover = HoverTool(tooltips=tooltips)

    # NOTE: Raster uses expected origin and axis limits; Image doesn't
    img = (
        hv.Raster(mat)
        .opts(
            title=f"Connectome [{start}, {stop})",
            cmap=CMAP,
            colorbar=True,
            xaxis=None,
            yaxis=None,
            frame_height=HEIGHT,
            data_aspect=1.0,
            tools=[hover],
        )
        .redim(z={"range": (-vmax, vmax)})
    )
    return img


def plot_carpet(path: str):
    timeseries = get_timeseries(path)

    tooltips = [
        ("(i, t)", "($y{0}, $x{0})"),
        ("value", "@image{0.00}"),
    ]
    hover = HoverTool(tooltips=tooltips)

    img = (
        hv.Image(
            np.flipud(timeseries),
            bounds=(0, 0, timeseries.shape[1], timeseries.shape[0]),
            kdims=["TR", "ROI"],
        )
        .opts(
            title="Timeseries",
            cmap=CMAP,
            colorbar=True,
            frame_height=HEIGHT,
            responsive=True,
            tools=[hover],
        )
        .redim(z={"range": (-3, 3)})
    )
    return img


def plot_window(path: str, frame: int, width: int):
    timeseries = get_timeseries(path)
    start, stop = get_window(frame, width, timeseries.shape[1])

    # NOTE: also tried Box and Rectangles. Box didn't allow filled color.
    # Rectangles was nice, but the fixed height was hard to work with. The
    # infinite vertical span here is awkward but oh well.
    span = hv.VSpan(start, stop).opts(color="gray", alpha=0.5)
    return span


def plot_lines(path: str, ranges: str):
    timeseries = get_timeseries(path)
    indices = parse_ranges(ranges)
    curve_dict = {idx: plot_line(timeseries, idx) for idx in indices}
    overlay = hv.NdOverlay(curve_dict, kdims=["ROI"])
    return overlay


def parse_ranges(ranges: str):
    indices = []
    for item in ranges.split(","):
        if "-" in item:
            start, stop = item.split("-")
        else:
            start, stop = item, None

        try:
            start = int(start.strip())
        except ValueError:
            continue

        stop = start if stop is None else int(stop.strip())
        indices.extend(list(range(start, stop + 1)))
    return indices


def plot_line(timeseries: np.ndarray, index: int):
    x = np.arange(timeseries.shape[1])
    y = timeseries[index]
    line = hv.Curve((x, y), kdims=["TR"], vdims=["Value"]).opts(
        xaxis=None,
        frame_height=(HEIGHT // 3),
        responsive=True,
        alpha=0.7,
        tools=["hover"],
    )
    return line


def plot_rms(path: str):
    timeseries = get_timeseries(path)
    x = np.arange(timeseries.shape[1])
    y = np.sqrt(np.mean(timeseries**2, axis=0))
    rms = hv.Curve((x, y), kdims=["TR"], vdims=["RMS"]).opts(
        frame_height=(HEIGHT // 3),
        responsive=True,
        color="black",
        alpha=0.7,
        tools=["hover"],
    )
    return rms


if __name__ == "__main__":
    parser = ArgumentParser("timeseries_viewer")
    parser.add_argument("--port", "-p", type=int, default=8310, help="App port")
    parser.add_argument("paths", nargs="+", help="Timeseries path(s)")
    args = parser.parse_args()
    main(args)
