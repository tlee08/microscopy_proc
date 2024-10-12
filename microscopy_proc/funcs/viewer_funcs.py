import functools
import os
from multiprocessing import Process

import dask.array as da
import napari
import tifffile
from dask.distributed import LocalCluster

from microscopy_proc.utils.dask_utils import cluster_proc_contxt
from microscopy_proc.utils.misc_utils import dictlists2listdicts
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model


# @flow
def view_arrs(fp_ls: tuple[str, ...], trimmer: tuple[slice, ...], **kwargs):
    with cluster_proc_contxt(LocalCluster()):
        # # OPTIONAL colourmaps
        # cmap_ls = ["gray", "green", "yellow"]
        # Reading arrays
        arr_ls = []
        for i in fp_ls:
            if ".zarr" in i:
                arr_ls.append(da.from_zarr(i)[*trimmer].compute())
            elif ".tif" in i:
                arr_ls.append(tifffile.imread(i)[*trimmer])
        # Asserting all kwargs_ls list lengths are equal to fp_ls length
        for k, v in kwargs.items():
            assert len(v) == len(fp_ls)
        # "Transposing" kwargs from dict of lists to list of dicts
        kwargs_ls = dictlists2listdicts(kwargs)
        # Making napari viewer
        viewer = napari.Viewer()
        # Adding image to napari viewer
        for i, arr in enumerate(arr_ls):
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

    imgs_ls = (
        # # ATLAS
        # ("ref", (0, 10000)),
        # ("annot", (0, 10000)),
        # # RAW
        # ("raw", (0, 7000)),
        # # REG
        # ("downsmpl_1", (0, 10000)),
        # ("downsmpl_2", (0, 10000)),
        ("trimmed", (0, 4000)),
        ("regresult", (0, 500)),
        # # MASK
        # ("mask", (0, 5)),
        # ("outline", (0, 5)),
        # ("mask_reg", (0, 5)),
        # # CELLC
        # ("overlap", (0, 10000)),
        # ("bgrm", (0, 2000)),
        # ("dog", (0, 100)),
        # ("adaptv", (0, 100)),
        # ("threshd", (0, 5)),
        # ("threshd_volumes", (0, 10000)),
        # ("threshd_filt", (0, 5)),
        ("maxima", (0, 5)),
        # ("wshed_volumes", (0, 1000)),
        ("wshed_filt", (0, 1000)),
        # # CELLC FINAL
        # ("threshd_final", (0, 5)),
        # ("maxima_final", (0, 5)),
        # ("wshed_final", (0, 1000)),
        # # POST
        # ("points_check", (0, 5)),
        # ("heatmap_check", (0, 20)),
        # ("points_trfm_check", (0, 5)),
        # ("heatmap_trfm_check", (0, 100)),
    )

    view_arrs(
        fp_ls=tuple(getattr(pfm, i) for i, j in imgs_ls),
        trimmer=trimmer,
        name=tuple(i for i, j in imgs_ls),
        contrast_limits=tuple(j for i, j in imgs_ls),
    )
