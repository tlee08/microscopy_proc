# %%


import logging

import dask.array as da
import napari
import tifffile
from dask.distributed import LocalCluster

from microscopy_proc.utils.dask_utils import cluster_proc_contxt
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


# @flow
def view_imgs(fp_ls, vmax_ls, slicer):
    with cluster_proc_contxt(LocalCluster()):
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
    # Running viewer
    napari.run()


if __name__ == "__main__":
    # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images/G3_agg_2.5x_1xzoom_03072024"

    proj_fp_dict = get_proj_fp_dict(proj_dir)

    slicer = (
        # slice(400, 500, None),  #  slice(None, None, 3),
        # slice(1000, 3000, None),  #  slice(None, None, 12),
        # slice(1000, 3000, None),  #  slice(None, None, 12),
        slice(600, 650, None),
        slice(2300, 3100, None),
        slice(2300, 3100, None),
        # slice(None, None, None),
        # slice(None, None, None),
        # slice(None, None, None),
    )

    imgs_ls = (
        # ATLAS
        # ("ref", 10000),
        # ("annot", 10000),
        # RAW
        # ("raw", 10000),
        # REG
        # ("downsmpl_1", 10000),
        # ("downsmpl_2", 10000),
        # ("trimmed", 10000),
        # ("regresult", 10000),
        # MASK
        # ("mask", 5),
        # ("outline", 5),
        # ("mask_reg", 5),
        # CELLC
        ("overlap", 10000),
        # ("bgrm", 2000),
        # ("dog", 100),
        ("adaptv", 100),
        # ("threshd", 5),
        ("threshd_sizes", 10000),
        # ("threshd_filt", 5),
        ("maxima", 5),
        ("wshed_sizes", 1000),
        ("wshed_filt", 1000),
        # CELLC FINAL
        # ("filt_final", 5),
        # ("maxima_final", 5),
        # ("wshed_sizes_final", 1000),
        # POST
        # ("points_check", 5),
        # ("heatmap_check", 20),
        # ("points_trfm_check", 5),
        # ("heatmap_trfm_check", 100),
    )
    fp_ls = [proj_fp_dict[i] for i, j in imgs_ls]
    vmax_ls = [j for i, j in imgs_ls]

    view_imgs(fp_ls, vmax_ls, slicer)
