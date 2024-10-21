import functools
import logging
import os
import re
from enum import Enum
from multiprocessing import Process
from typing import Optional

import dask.array as da
import napari
import numpy as np
import tifffile
from dask.distributed import LocalCluster

from microscopy_proc.utils.dask_utils import cluster_proc_contxt
from microscopy_proc.utils.misc_utils import dictlists2listdicts
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

VRANGE = "vrange"
CMAP = "cmap"


class Colormaps(Enum):
    GRAY = "gray"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    SET1 = "Set1"


IMGS = {
    "Atlas": {
        "ref": {VRANGE: (0, 10000), CMAP: Colormaps.GREEN.value},
        "annot": {VRANGE: (0, 10000), CMAP: Colormaps.SET1.value},
    },
    "Raw": {
        "raw": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
    },
    "Registration": {
        "downsmpl1": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
        "downsmpl2": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
        "trimmed": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
        "regresult": {VRANGE: (0, 1000), CMAP: Colormaps.GREEN.value},
    },
    "Mask": {
        "premask_blur": {VRANGE: (0, 10000), CMAP: Colormaps.RED.value},
        "mask": {VRANGE: (0, 5), CMAP: Colormaps.RED.value},
        "outline": {VRANGE: (0, 5), CMAP: Colormaps.RED.value},
        "mask_reg": {VRANGE: (0, 5), CMAP: Colormaps.RED.value},
    },
    "Cell Counting (overlapped)": {
        "overlap": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
        "bgrm": {VRANGE: (0, 2000), CMAP: Colormaps.GREEN.value},
        "dog": {VRANGE: (0, 500), CMAP: Colormaps.RED.value},
        "adaptv": {VRANGE: (0, 500), CMAP: Colormaps.RED.value},
        "threshd": {VRANGE: (0, 5), CMAP: Colormaps.GRAY.value},
        "threshd_volumes": {VRANGE: (0, 10000), CMAP: Colormaps.GREEN.value},
        "threshd_filt": {VRANGE: (0, 5), CMAP: Colormaps.GREEN.value},
        "maxima": {VRANGE: (0, 5), CMAP: Colormaps.GREEN.value},
        "wshed_volumes": {VRANGE: (0, 1000), CMAP: Colormaps.GREEN.value},
        "wshed_filt": {VRANGE: (0, 1000), CMAP: Colormaps.GREEN.value},
    },
    "Cell Counting (trimmed)": {
        "threshd_final": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
        "maxima_final": {VRANGE: (0, 5), CMAP: Colormaps.RED.value},
        "wshed_final": {VRANGE: (0, 1000), CMAP: Colormaps.GREEN.value},
    },
    "Post Processing Checks": {
        "points_check": {VRANGE: (0, 5), CMAP: Colormaps.GREEN.value},
        "heatmap_check": {VRANGE: (0, 20), CMAP: Colormaps.RED.value},
        "points_trfm_check": {VRANGE: (0, 5), CMAP: Colormaps.GREEN.value},
        "heatmap_trfm_check": {VRANGE: (0, 100), CMAP: Colormaps.RED.value},
    },
}


def read_img(fp, trimmer: Optional[tuple[slice, ...]] = None):
    """
    Reading, trimming (if possible), and returning the array in memory.
    """
    if re.search(r"\.zarr$", fp):
        return da.from_zarr(fp)[*trimmer].compute()
    elif re.search(r"\.tif$", fp):
        return tifffile.imread(fp)[*trimmer]
    else:
        raise NotImplementedError("Only .zarr and .tif files are supported.")


def view_arrs(fp_ls: tuple[str, ...], trimmer: tuple[slice, ...], **kwargs):
    with cluster_proc_contxt(LocalCluster()):
        # Asserting all kwargs_ls list lengths are equal to fp_ls length
        for k, v in kwargs.items():
            assert len(v) == len(fp_ls)
        # Reading arrays
        arr_ls = []
        for i, fp in enumerate(fp_ls):
            logging.info(f"Loading image # {i} / {len(arr_ls)}")
            arr_ls.append(read_img(fp, trimmer))
        # "Transposing" kwargs from dict of lists to list of dicts
        kwargs_ls = dictlists2listdicts(kwargs)
        # Making napari viewer
        viewer = napari.Viewer()
        # Adding image to napari viewer
        for i, arr in enumerate(arr_ls):
            logging.info(f"Napari viewer - adding image # {i} / {len(arr_ls)}")
            # NOTE: best kwargs to use are name, contrast_limits, and colormap
            viewer.add_image(
                data=arr,
                blending="additive",
                **kwargs_ls[i],
            )
    # Running viewer
    napari.run()


@functools.wraps(view_arrs)
def view_arrs_mp(fp_ls: tuple[str, ...], trimmer: tuple[slice, ...], **kwargs):
    logging.info("Starting napari viewer")
    # Making napari viewer multiprocess
    napari_proc = Process(target=view_arrs, args=(fp_ls, trimmer), kwargs=kwargs)
    napari_proc.start()
    # napari_proc.join()


# TODO: implement elsewhere for usage examples
def save_arr(
    fp_in: str,
    fp_out: str,
    trimmer: Optional[tuple[slice, ...]] = None,
    **kwargs,
):
    """
    NOTE: exports as tiff only.
    """
    with cluster_proc_contxt(LocalCluster()):
        # Reading
        arr = read_img(fp_in, trimmer)
        # Writing
        tifffile.imwrite(fp_out, arr)


def combine_arrs(
    fp_in_ls: tuple[str, ...],
    fp_out: str,
    trimmer: Optional[tuple[slice, ...]] = None,
    **kwargs,
):
    dtype = np.uint16
    # Reading arrays
    arr_ls = []
    for i in fp_in_ls:
        # Read image
        arr = read_img(i, trimmer)
        # Convert to dtype (rounded and within clip bounds)
        arr = arr.round(0).clip(0, 2**16 - 1).astype(dtype)
        # Adding image to list
        arr_ls.append(arr)
    # Stacking arrays
    arr = np.stack(arr_ls, axis=-1, dtype=dtype)
    # Writing to file
    tifffile.imwrite(fp_out, arr)


if __name__ == "__main__":
    # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    proj_dir = os.path.join(proj_dir, "P15_2.5x_1x_zoom_07082024")

    proj_dir = "/home/linux1/Desktop/example_proj"

    pfm = get_proj_fp_model(proj_dir)

    trimmer = (
        slice(600, 650, None),
        slice(1400, 3100, None),
        slice(500, 3100, None),
        # slice(None, None, None),
        # slice(None, None, None),
        # slice(None, None, None),
    )

    imgs_to_run_dict = {
        "Atlas": [
            "ref",
            "annot",
        ],
        "Raw": [
            "raw",
        ],
        "Registration": [
            "downsmpl1",
            "downsmpl2",
            "trimmed",
            "regresult",
        ],
        "Mask": [
            "premask_blur",
            "mask",
            "outline",
            "mask_reg",
        ],
        "Cell Counting (overlapped)": [
            "overlap",
            "bgrm",
            "dog",
            "adaptv",
            "threshd",
            "threshd_volumes",
            "threshd_filt",
            "maxima",
            "wshed_volumes",
            "wshed_filt",
        ],
        "Cell Counting (trimmed)": [
            "threshd_final",
            "maxima_final",
            "wshed_final",
        ],
        "Post Processing Checks": [
            "points_check",
            "heatmap_check",
            "points_trfm_check",
            "heatmap_trfm_check",
        ],
    }

    fp_ls = []
    name = []
    contrast_limits = []
    colormap = []
    for group_k, group_v in imgs_to_run_dict.items():
        for img_i in group_v:
            fp_ls.append(getattr(pfm, img_i))
            name.append(img_i)
            contrast_limits.append(IMGS[group_k][img_i][VRANGE])
            colormap.append(IMGS[group_k][img_i][CMAP])

    view_arrs(
        fp_ls=tuple(fp_ls),
        trimmer=trimmer,
        name=tuple(name),
        contrast_limits=tuple(contrast_limits),
        colormap=tuple(colormap),
    )
