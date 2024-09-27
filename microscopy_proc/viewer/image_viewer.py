import logging
import os

import dask.array as da
import napari
import tifffile
from dask.distributed import LocalCluster

from microscopy_proc.utils.dask_utils import cluster_proc_contxt
from microscopy_proc.utils.proj_org_utils import get_proj_fp_dict


# @flow
def view_imgs(fp_ls, vmax_ls, slicer):
    with cluster_proc_contxt(LocalCluster()):
        # OPTIONAL colourmaps
        cmap_ls = ["gray", "green", "yellow"]
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
        for i, arr in enumerate(arr_ls):
            vmax = vmax_ls[i]
            cmap = cmap_ls[i] if i < len(cmap_ls) else "gray"
            viewer.add_image(
                arr,
                # name=arr.__name__,
                contrast_limits=(0, vmax),
                blending="additive",
                colormap=cmap,
            )
    # Running viewer
    napari.run()


if __name__ == "__main__":
    # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    proj_dir = os.path.join(proj_dir, "P8_2.5x_1x_zoom_07082024")

    proj_fp_dict = get_proj_fp_dict(proj_dir)

    slicer = (
        # slice(400, 500, None),  #  slice(None, None, 3),
        # slice(1000, 3000, None),  #  slice(None, None, 12),
        # slice(1000, 3000, None),  #  slice(None, None, 12),
        # slice(600, 650, None),
        # slice(2300, 3100, None),
        # slice(2300, 3100, None),
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None),
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
        ("trimmed", 4000),
        ("regresult", 500),
        # MASK
        # ("mask", 5),
        # ("outline", 5),
        # ("mask_reg", 5),
        # CELLC
        # ("overlap", 10000),
        # ("bgrm", 2000),
        # ("dog", 100),
        # ("adaptv", 100),
        # ("threshd", 5),
        # ("threshd_sizes", 10000),
        # ("threshd_filt", 5),
        # ("maxima", 5),
        # ("wshed_sizes", 1000),
        # ("wshed_filt", 1000),
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
