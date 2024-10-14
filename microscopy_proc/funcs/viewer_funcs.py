import functools
import logging
import os
from enum import Enum
from multiprocessing import Process

import dask.array as da
import napari
import tifffile
from dask.distributed import LocalCluster

from microscopy_proc.utils.dask_utils import cluster_proc_contxt
from microscopy_proc.utils.misc_utils import dictlists2listdicts
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

VIEW = "viewer"
VRANGE_D = f"{VIEW}_vrange_default"
CMAP_D = f"{VIEW}_cmap_default"


class Colormaps(Enum):
    GRAY = "gray"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    SET1 = "Set1"


VIEWER_IMGS = {
    "Atlas": {
        "ref": {VRANGE_D: (0, 10000), CMAP_D: Colormaps.GREEN.value},
        "annot": {VRANGE_D: (0, 10000), CMAP_D: Colormaps.SET1.value},
    },
    "Raw": {
        "raw": {VRANGE_D: (0, 10000), CMAP_D: Colormaps.GRAY.value},
    },
    "Registration": {
        "downsmpl1": {VRANGE_D: (0, 10000), CMAP_D: Colormaps.GRAY.value},
        "downsmpl2": {VRANGE_D: (0, 10000), CMAP_D: Colormaps.GRAY.value},
        "trimmed": {VRANGE_D: (0, 10000), CMAP_D: Colormaps.GRAY.value},
        "regresult": {VRANGE_D: (0, 1000), CMAP_D: Colormaps.GREEN.value},
    },
    "Mask": {
        "premask_blur": {VRANGE_D: (0, 10000), CMAP_D: Colormaps.RED.value},
        "mask": {VRANGE_D: (0, 5), CMAP_D: Colormaps.RED.value},
        "outline": {VRANGE_D: (0, 5), CMAP_D: Colormaps.RED.value},
        "mask_reg": {VRANGE_D: (0, 5), CMAP_D: Colormaps.RED.value},
    },
    "Cell Counting (overlapped)": {
        "overlap": {VRANGE_D: (0, 10000), CMAP_D: Colormaps.GRAY.value},
        "bgrm": {VRANGE_D: (0, 2000), CMAP_D: Colormaps.GREEN.value},
        "dog": {VRANGE_D: (0, 100), CMAP_D: Colormaps.RED.value},
        "adaptv": {VRANGE_D: (0, 100), CMAP_D: Colormaps.RED.value},
        "threshd": {VRANGE_D: (0, 5), CMAP_D: Colormaps.GRAY.value},
        "threshd_volumes": {VRANGE_D: (0, 10000), CMAP_D: Colormaps.GREEN.value},
        "threshd_filt": {VRANGE_D: (0, 5), CMAP_D: Colormaps.GREEN.value},
        "maxima": {VRANGE_D: (0, 5), CMAP_D: Colormaps.GREEN.value},
        "wshed_volumes": {VRANGE_D: (0, 1000), CMAP_D: Colormaps.GREEN.value},
        "wshed_filt": {VRANGE_D: (0, 1000), CMAP_D: Colormaps.GREEN.value},
    },
    "Cell Counting (trimmed)": {
        "threshd_final": {VRANGE_D: (0, 10000), CMAP_D: Colormaps.GRAY.value},
        "maxima_final": {VRANGE_D: (0, 5), CMAP_D: Colormaps.RED.value},
        "wshed_final": {VRANGE_D: (0, 1000), CMAP_D: Colormaps.GREEN.value},
    },
    "Post Processing Checks": {
        "points_check": {VRANGE_D: (0, 5), CMAP_D: Colormaps.GREEN.value},
        "heatmap_check": {VRANGE_D: (0, 20), CMAP_D: Colormaps.RED.value},
        "points_trfm_check": {VRANGE_D: (0, 5), CMAP_D: Colormaps.GREEN.value},
        "heatmap_trfm_check": {VRANGE_D: (0, 100), CMAP_D: Colormaps.RED.value},
    },
}


def view_arrs(fp_ls: tuple[str, ...], trimmer: tuple[slice, ...], **kwargs):
    with cluster_proc_contxt(LocalCluster()):
        # Asserting all kwargs_ls list lengths are equal to fp_ls length
        for k, v in kwargs.items():
            assert len(v) == len(fp_ls)
        # Reading arrays
        arr_ls = []
        for i, fp in enumerate(fp_ls):
            logging.info(f"Loading image # {i} / {len(arr_ls)}")
            if ".zarr" in fp:
                arr_ls.append(da.from_zarr(fp)[*trimmer].compute())
            elif ".tif" in fp:
                arr_ls.append(tifffile.imread(fp)[*trimmer])
        # "Transposing" kwargs from dict of lists to list of dicts
        kwargs_ls = dictlists2listdicts(kwargs)
        # Making napari viewer
        viewer = napari.Viewer()
        # Adding image to napari viewer
        for i, arr in enumerate(arr_ls):
            logging.info(f"Napari viewer - adding image # {i} / {len(arr_ls)}")
            viewer.add_image(
                data=arr,
                blending="additive",
                **kwargs_ls[i],
                # name=ar.__name__,
                # contrast_limits=(0, vmax),
                # colormap=cmap,
            )
    # Running viewer
    napari.run()


@functools.wraps(view_arrs)
def view_arrs_mp(fp_ls: tuple[str, ...], trimmer: tuple[slice, ...], **kwargs):
    logging.info("Starting napari viewer")
    napari_proc = Process(target=view_arrs, args=(fp_ls, trimmer), kwargs=kwargs)
    napari_proc.start()
    # napari_proc.join()


if __name__ == "__main__":
    # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    proj_dir = os.path.join(proj_dir, "P15_2.5x_1x_zoom_07082024")

    proj_dir = "/home/linux1/Desktop/example_proj"

    pfm = get_proj_fp_model(proj_dir)

    trimmer = (
        # slice(400, 500, None),  #  slice(None, None, 3),
        # slice(1000, 3000, None),  #  slice(None, None, 12),
        # slice(1000, 3000, None),  #  slice(None, None, 12),
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
            contrast_limits.append(VIEWER_IMGS[group_k][img_i][VRANGE_D])
            colormap.append(VIEWER_IMGS[group_k][img_i][CMAP_D])

    view_arrs(
        fp_ls=tuple(fp_ls),
        trimmer=trimmer,
        name=tuple(name),
        contrast_limits=tuple(contrast_limits),
        colormap=tuple(colormap),
    )
