# %%


import logging

import dask.array as da
import napari
import tifffile
from dask.distributed import LocalCluster

from microscopy_proc.utils.dask_utils import cluster_proc_dec
from microscopy_proc.utils.proj_org_utils import get_proj_fp_dict

# %%


# @task
def add_img(viewer, arr, vmax):
    viewer.add_image(
        arr,
        # name=arr.__name__,
        contrast_limits=(0, vmax),
        blending="additive",
    )


@cluster_proc_dec(lambda: LocalCluster())
# @flow
def view_imgs(fp_ls, vmax_ls, slicer):
    # Reading arrays
    arr_ls = []
    for i in fp_ls:
        logging.info(i)
        if ".zarr" in i:
            arr_ls.append(da.from_zarr(i)[*slicer].compute())
        elif ".tif" in i:
            arr_ls.append(tifffile.imread(i)[*slicer])

    # Napari viewer adding images
    viewer = napari.Viewer()
    for i, j in zip(arr_ls, vmax_ls):
        add_img(viewer, i, j)

    napari.run()


if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"

    proj_fp_dict = get_proj_fp_dict(proj_dir)

    slicer = (
        # slice(400, 500, None),  #  slice(None, None, 3),
        # slice(1000, 3000, None),  #  slice(None, None, 12),
        # slice(1000, 3000, None),  #  slice(None, None, 12),
        slice(200, 400, None),
        slice(1000, 4000, None),
        slice(2000, None, None),
        # slice(None, None, None),
        # slice(None, None, None),
        # slice(None, None, None),
    )

    imgs_ls = (
        # ("ref", 10000),
        # ("annot", 10000),
        # RAW
        ("raw", 10000),
        # REG
        # ("downsmpl_1", 10000),
        # ("downsmpl_2", 10000),
        # ("trimmed", 10000),
        # ("regresult", 10000),
        # CELLC
        # ("overlap", 10000),
        # ("bgrm", 2000),
        # ("dog", 100),
        # ("adaptv", 100),
        # ("threshd", 5),
        # ("sizes", 10000),
        # ("filt", 5),
        # ("maxima", 5),
        # ("filt_final", 5),
        # ("maxima_final", 2),
        # POST
        # ("points_check", 5),
        # ("heatmap_check", 20),
        # ("points_trfm_check", 5),
        # ("heatmap_trfm_check", 100),
    )
    fp_ls = [proj_fp_dict[i] for i, j in imgs_ls]
    vmax_ls = [j for i, j in imgs_ls]

    view_imgs(fp_ls, vmax_ls, slicer)
