"""
A pyviz panel/holoviews app for visualizing fMRI timeseries.
"""

import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Union

import holoviews as hv
import numpy as np
import panel as pn
from bokeh.models import HoverTool
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

from neuro_dashboards.utils.parcellations import Parcellation, get_schaefer

logging.basicConfig(level=logging.INFO)
hv.extension("bokeh")

# Needed to get long filenames to scroll in multiselect
css = """
.bk.panel-multiselect * {
    overflow-x: scroll;
}
"""
pn.extension(sizing_mode="stretch_width", raw_css=[css])

HEIGHT = 300
CMAP = LinearSegmentedColormap.from_list(
    "conmat", ["blue", "cyan", "white", "yellow", "red"]
)
AVAILABLE_PARCS = [f"schaefer{parcels}" for parcels in range(100, 1100, 100)]
CACHE = {}


@dataclass
class Args:
    parc: str
    port: int
    paths: List[str]


def get_parser() -> ArgumentParser:
    parser = ArgumentParser("timeseries_viewer")
    parser.add_argument(
        "--parc",
        type=str,
        default=None,
        choices=AVAILABLE_PARCS,
        help="Name of parcellation",
    )
    parser.add_argument("--port", type=int, default=8310, help="App port")
    parser.add_argument("paths", nargs="+", help="Timeseries path(s)")
    return parser


def main(args: Args):
    if args.parc is not None:
        CACHE["parc"] = get_parcellation(args.parc)
    timeseries, _ = get_timeseries(args.paths[:1])

    path_select = pn.widgets.MultiSelect(
        name="TS path(s)",
        value=args.paths[:1],
        options={Path(p).name: p for p in args.paths},
        size=16,
        css_classes=["panel-multiselect"],
    )
    frame_slider = pn.widgets.IntSlider(
        name="Time",
        value=0,
        start=0,
        end=(timeseries.shape[1] - 1),
    )
    width_slider = pn.widgets.IntSlider(
        name="Width",
        value=16,
        start=1,
        step=2,
        end=timeseries.shape[1],
    )
    backward = pn.widgets.Button(name="\u25c0", width=50)
    forward = pn.widgets.Button(name="\u25b6", width=50)
    ranges_input = pn.widgets.TextInput(
        name="Timeseries ranges", placeholder="0,1,3-5", value="0"
    )

    grad_options = {"PC1": 1, "PC2": 2, "PC3": 3, "None": -1}
    grad_select = pn.widgets.Select(name="Gradient", value=1, options=grad_options)

    def reset_selection(event):
        timeseries, _ = get_timeseries(event.new)
        frame_slider.end = timeseries.shape[1] - 1
        width_slider.end = timeseries.shape[1]
        frame_slider.value = 0
        width_slider.value = 16

    path_select.param.watch(reset_selection, ["value"])

    def shift_back(event):
        step = max(width_slider.value // 3, 1)
        frame = max(frame_slider.start, frame_slider.value - step)
        frame_slider.value = frame

    def shift_forward(event):
        step = max(width_slider.value // 3, 1)
        frame = min(frame_slider.end, frame_slider.value + step)
        frame_slider.value = frame

    backward.on_click(shift_back)
    forward.on_click(shift_forward)

    carpet = pn.depends(paths=path_select)(plot_carpet)
    conmat = pn.depends(paths=path_select, frame=frame_slider, width=width_slider)(
        plot_conmat
    )
    window = pn.depends(paths=path_select, frame=frame_slider, width=width_slider)(
        plot_window
    )
    breaks = pn.depends(paths=path_select)(plot_breaks)
    lines = pn.depends(paths=path_select, ranges=ranges_input)(plot_lines)
    rms = pn.depends(paths=path_select)(plot_rms)
    gradients = pn.depends(
        paths=path_select,
        frame=frame_slider,
        width=width_slider,
        selected_comp=grad_select,
    )(plot_gradients)

    # NOTE: framewise=True needed to get the axis ranges to update
    carpet_dmap = hv.DynamicMap(carpet).opts(framewise=True)
    conmat_dmap = hv.DynamicMap(conmat)
    window_dmap = hv.DynamicMap(window)
    breaks_dmap = hv.DynamicMap(breaks)
    lines_dmap = hv.DynamicMap(lines).opts(framewise=True)
    rms_dmap = hv.DynamicMap(rms).opts(framewise=True)
    gradients_dmap = hv.DynamicMap(gradients)

    app = pn.template.BootstrapTemplate(title="Timeseries viewer")
    app.sidebar.extend(
        [
            path_select,
            grad_select,
            ranges_input,
            frame_slider,
            width_slider,
            pn.Row(backward, forward),
        ]
    )

    ts_view = pn.Column(
        ((carpet_dmap * window_dmap * breaks_dmap) + conmat_dmap).opts(
            shared_axes=False
        ),
        (
            (lines_dmap * window_dmap * breaks_dmap)
            + (rms_dmap * window_dmap * breaks_dmap)
        ).cols(1),
    )
    grad_view = pn.Column(gradients_dmap, (rms_dmap * window_dmap * breaks_dmap))
    app.main.append(pn.Tabs(("Timeseries", ts_view), ("Gradients", grad_view)))
    pn.serve(app, port=args.port)


def get_timeseries(paths: List[str]):
    return _get_timeseries(tuple(paths))


@lru_cache(maxsize=1)
def _get_timeseries(paths: Tuple[str, ...]):
    logging.info(f"Loading timeseries\n\t{paths}")
    timeseries = [get_single_timeseries(p) for p in paths]
    breaks = np.cumsum(np.array([0] + [len(ts) for ts in timeseries]))
    timeseries = np.concatenate(timeseries)
    timeseries = np.ascontiguousarray(timeseries.T)
    return timeseries, breaks


def get_single_timeseries(path: Union[str, Path]):
    path = Path(path)
    timeseries: np.ndarray
    if path.suffix == ".tsv":
        # TODO: what is the shape of these files?
        timeseries = np.loadtxt(path, delimiter="\t", dtype=float)
    elif path.suffix == ".npy":
        timeseries = np.load(path)
    else:
        raise ValueError(f"Invalid path {path}; expected .tsv or .npy")
    timeseries -= timeseries.mean(axis=0)
    timeseries /= timeseries.std(axis=0) + 1e-8
    return timeseries


def get_window(frame: int, width: int, length: int):
    start = max(frame - width // 2, 0)
    stop = min(start + width, length)
    return start, stop


def plot_conmat(paths: List[str], frame: int, width: int):
    timeseries, _ = get_timeseries(paths)

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

    # restricting zoom out
    # https://github.com/holoviz/holoviews/issues/1019
    def set_bounds(fig, _):
        fig.state.x_range.bounds = (0, mat.shape[1])
        fig.state.y_range.bounds = (0, mat.shape[0])

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
            hooks=[set_bounds],
        )
        .redim(z={"range": (-vmax, vmax)})
    )
    return img


def plot_carpet(paths: List[str]):
    timeseries, _ = get_timeseries(paths)

    tooltips = [
        ("(i, t)", "($y{0}, $x{0})"),
        ("value", "@image{0.00}"),
    ]
    hover = HoverTool(tooltips=tooltips)

    # restricting zoom out
    # https://github.com/holoviz/holoviews/issues/1019
    def set_bounds(fig, _):
        fig.state.x_range.bounds = (0, timeseries.shape[1])
        fig.state.y_range.bounds = (0, timeseries.shape[0])

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
            hooks=[set_bounds],
            xlim=(0, timeseries.shape[1]),
        )
        .redim(z={"range": (-3, 3)})
    )
    return img


def plot_window(paths: List[str], frame: int, width: int):
    timeseries, _ = get_timeseries(paths)
    start, stop = get_window(frame, width, timeseries.shape[1])

    # NOTE: also tried Box and Rectangles. Box didn't allow filled color.
    # Rectangles was nice, but the fixed height was hard to work with. The
    # infinite vertical span here is awkward but oh well.
    span = hv.VSpan(start, stop).opts(color="gray", alpha=0.5)
    return span


def plot_breaks(paths: List[str]):
    _, breaks = get_timeseries(paths)
    lines = {brk: hv.VLine(brk).opts(color="black", alpha=0.5) for brk in breaks}
    overlay = hv.NdOverlay(lines)
    return overlay


def plot_lines(paths: List[str], ranges: str):
    timeseries, _ = get_timeseries(paths)
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
        xlim=(0, timeseries.shape[1]),
    )
    return line


def plot_rms(paths: List[str]):
    timeseries, _ = get_timeseries(paths)
    x = np.arange(timeseries.shape[1])
    y = np.sqrt(np.mean(timeseries**2, axis=0))
    rms = hv.Curve((x, y), kdims=["TR"], vdims=["RMS"]).opts(
        frame_height=(HEIGHT // 3),
        responsive=True,
        color="black",
        alpha=0.7,
        tools=["hover"],
        xlim=(0, timeseries.shape[1]),
    )
    return rms


def plot_gradients(
    paths: List[str],
    frame: int,
    width: int,
    selected_comp: int,
):
    parc: Parcellation = CACHE.get("parc")

    if parc is None or selected_comp < 0:
        # Dummy plot, much faster to render.
        # NOTE: you can't plot this first.
        # NOTE: struggled to make these plots responsive.
        # TODO: what if you reload the page?
        img = hv.Polygons([], vdims=["value", "name", "index"]).opts(
            title="None",
            cmap=CMAP,
            colorbar=True,
            xaxis=None,
            yaxis=None,
            tools=["hover"],
            toolbar="above",
        )
        return img

    gradients = get_principal_gradients(tuple(paths), frame, width, selected_comp)
    gradient = gradients[-1]
    vmax = np.quantile(np.abs(gradient), 0.95)

    poly_list = parc.poly_list(values=gradient)
    img = (
        hv.Polygons(poly_list, vdims=["value", "name", "index"])
        .opts(
            title="Mean" if selected_comp == 0 else f"PC{selected_comp}",
            cmap=CMAP,
            colorbar=True,
            xaxis=None,
            yaxis=None,
            data_aspect=1.0,
            tools=["hover"],
            toolbar="above",
        )
        .redim(value={"range": (-vmax, vmax)})
    )
    return img


@lru_cache(maxsize=128)
def get_principal_gradients(
    paths: Tuple[str, ...],
    frame: int,
    width: int,
    n_components: int,
) -> np.ndarray:

    timeseries, _ = get_timeseries(paths)
    start, stop = get_window(frame, width, timeseries.shape[1])
    window = np.ascontiguousarray(timeseries[:, start:stop].T)

    gradients = np.full((n_components + 1, window.shape[1]), np.nan)
    valid_nc = min(len(window) - 1, n_components)
    if valid_nc == 0:
        gradients[0] = window.mean(axis=0)
    else:
        embed = PCA(n_components=valid_nc, svd_solver="randomized", random_state=42)
        embed.fit(window)
        gradients[0] = embed.mean_
        gradients[1:] = embed.singular_values_[:, None] * embed.components_

        # deal with sign flips; hack
        gradients[1:] *= np.sign(np.sum(gradients[1:], axis=1, keepdims=True))
    return gradients


def get_parcellation(name: str) -> Parcellation:
    logging.info(f"Loading parcellation {name}")
    name = name.lower()

    # schaefer200
    if name.startswith("schaefer"):
        parcels = int(name[len("schaefer") :])
        parc = get_schaefer(parces=parcels)
    else:
        raise NotImplementedError(f"Parcellation {name} not supported")

    # pre-compute ROI polys
    parc.poly_list()
    return parc


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
