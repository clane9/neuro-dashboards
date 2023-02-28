import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve

import nibabel as nib
import numpy as np
from shapely import MultiPolygon

from .surface import Surface, load_fsaverage_flat

CACHE_DIR = Path.home() / ".cache/parcellations"


@dataclass
class Parcellation:
    # ROI label array where 0 is background
    label: np.ndarray
    # mapping from ROI indices to names
    names: Dict[int, str]
    # mapping from ROI indices to RGB colors
    colors: Dict[int, Tuple[int, int, int]]
    surf: Surface

    def __post_init__(self):
        assert len(self.surf) == len(self.label), "Parcellation doesn't match surface"
        self._polys: Optional[Dict[int, MultiPolygon]] = None
        self._poly_list: Optional[List[Dict[str, Any]]] = None

    @property
    def polys(self) -> Dict[int, MultiPolygon]:
        """
        Mapping of ROI IDs to polygon boundaries.
        """
        if self._polys is None:
            self._polys = {
                idx: self.surf.roi_to_poly(
                    mask=(self.label == idx), simplify_tolerance=1.0
                )
                for idx in self.names
            }
        return self._polys

    def poly_list(self, values: Optional[np.ndarray] = None):
        """
        List of ROI polygon records compatible with holoviews.
        """
        if self._poly_list is None:
            poly_list = []

            for idx, multipoly in self.polys.items():
                for poly in multipoly.geoms:
                    xy = np.asarray(poly.exterior.coords)
                    poly_dict = {
                        "x": xy[:, 0],
                        "y": xy[:, 1],
                        "index": idx,
                        "name": self.names[idx],
                        "value": float("nan"),
                    }
                    poly_list.append(poly_dict)
            self._poly_list = poly_list

        poly_list = self._poly_list
        if values is not None:
            assert len(values) == len(self), "Values don't match parcellation"
            poly_list = [
                self._with_update_value(record, values) for record in poly_list
            ]
        return poly_list

    @staticmethod
    def _with_update_value(record: Dict[str, Any], values: np.ndarray):
        idx = record["index"]
        if idx <= 0:
            return record
        record = record.copy()
        record["value"] = values[idx - 1]
        return record

    def __len__(self) -> int:
        return len(self.names)


def get_schaefer(
    parcels: int = 200,
    networks: int = 7,
    **kwargs,
) -> Parcellation:

    cache_path = (
        CACHE_DIR / f"Schaefer2018_{parcels}Parcels_{networks}Networks_order.parc.pkl"
    )
    if cache_path.exists():
        with cache_path.open("rb") as f:
            parc = pickle.load(f)
            return parc

    label, colors, names = None, None, None
    for hemi in ["lh", "rh"]:
        filename, url = _get_schaefer_url(hemi=hemi, parcels=parcels, networks=networks)
        path = _maybe_download(filename, url)
        hemi_label, hemi_colors, hemi_names = nib.freesurfer.read_annot(path)

        # Assume 0 = background; skip
        hemi_colors = hemi_colors[1:]
        hemi_names = hemi_names[1:]

        if label is None:
            label = hemi_label
            colors = hemi_colors
            names = hemi_names
        else:
            offset = label.max()
            hemi_label = np.where(hemi_label > 0, hemi_label + offset, 0)
            label = np.concatenate([label, hemi_label])
            colors = np.concatenate([colors, hemi_colors])
            names = np.concatenate([names, hemi_names])

    names = {idx + 1: _as_str(name) for idx, name in enumerate(names)}
    colors = {idx + 1: tuple(color[:3]) for idx, color in enumerate(colors)}
    surf = load_fsaverage_flat()
    parc = Parcellation(label, names, colors, surf)

    # Pre-compute polys
    parc.poly_list()

    with cache_path.open("wb") as f:
        pickle.dump(parc, f)
    return parc


def _get_schaefer_url(
    hemi: str = "lh", parcels: int = 200, networks: int = 7, **kwargs
) -> Tuple[str, str]:
    assert hemi in {"lh", "rh"}, f"Invalid hemi {hemi}"
    assert parcels in {ii * 100 for ii in range(1, 11)}, f"Invalid parcels {parcels}"
    assert networks in {7, 17}, f"Invalid networks {networks}"

    name = f"{hemi}.Schaefer2018_{parcels}Parcels_{networks}Networks_order.annot"
    url = (
        "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/"
        "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/"
        f"FreeSurfer5.3/fsaverage/label/{name}"
    )
    return name, url


def _maybe_download(
    name: str, url: str, cache_dir: Union[str, Path] = CACHE_DIR
) -> Path:
    path = Path(cache_dir) / name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)
    return path


def _as_str(s: Any) -> str:
    if isinstance(s, bytes):
        s = s.decode()
    return str(s)
